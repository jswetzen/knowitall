# Deploying knowitall to Proxmox (`mikro`) as IaC

> Initial Tofu config lives in [`deploy/proxmox/`](../deploy/proxmox/). This doc is the design; that directory is the implementation.

Target stack: OpenTofu provisions an unprivileged LXC on `mikro`, cloud-init bootstraps rootless Podman, a Quadlet `.container` unit runs the `knowitall` server. Backups via PBS. Encryption via host-level LUKS on the LVM thinpool. Per-container observability via a Pulse agent inside the LXC.

This is a niche combination — none of the homelab repos I surveyed do all four layers end-to-end, so expect to assemble rather than clone. See [Known gotchas](#known-gotchas) before you start; the overlayfs and AppArmor ones are where time gets lost.

## Why this shape

- **LXC over VM**: shared kernel, no virtualization overhead, individual container processes visible from the host. Trade-off accepted: weaker isolation boundary, kernel CVEs can cross the LXC.
- **Rootless Podman + Quadlet over Docker/Portainer**: declarative units in version control, proper systemd integration (journald, dependency ordering, cgroup accounting), no daemon. `podman generate systemd` is deprecated; `podman-compose` is explicitly not recommended for production by upstream.
- **Single `.container` unit**: knowitall is one service (`knowitall`) with embedded Kùzu + LanceDB stores as files under `/data`. No multi-container orchestration needed — Quadlet is a strict win over compose here.
- **Host-level LUKS on the thinpool**: unlocked once at Proxmox boot, LXC rootfs lives on it transparently. Simplest key management; acceptable threat model for a homelab where the concern is physical theft rather than a compromised running host.

## Architecture

```
Proxmox host (mikro)
├── LUKS-encrypted LVM thinpool         ← unlocked at boot
│   └── unprivileged LXC (Debian 13)
│       ├── nesting=1, keyctl=1, fuse=1
│       ├── subuid/subgid mapped
│       ├── rootless podman as service user
│       │   └── ~/.config/containers/systemd/knowitall.container  ← Quadlet
│       │       └── podman container: knowitall
│       │           ├── :8765 → host
│       │           └── /data → bind mount (kùzu + lancedb files)
│       └── Pulse agent (per-container metrics)
└── PBS backup target                   ← stop-then-snapshot, brief downtime
```

External dependency: Ollama at `192.168.1.33:11434` (already running elsewhere on the LAN).

## Provisioning layers

### 1. Proxmox host prep (one-time, manual)

- Create LUKS-encrypted LVM thinpool. Add the unlock to `/etc/crypttab` keyed to a file on an unencrypted boot partition, or accept manual unlock at boot. Document which choice you made — it affects unattended reboot recovery.
- Install PBS client; configure datastore + retention policy.
- Generate an API token for the bpg/proxmox provider with the minimum roles needed (`VM.Allocate`, `VM.Config.*`, `Datastore.AllocateSpace`, `SDN.Use` if relevant).

### 2. OpenTofu (the LXC)

Use the `bpg/proxmox` provider (v0.106.0+, actively maintained, Proxmox 9.x compatible). The `kode3tech/terraform-proxmox-lxc` module is the closest off-the-shelf option and already handles `nesting=true`, FUSE, and keyctl — the prereqs for Podman.

Key resource settings (`proxmox_virtual_environment_container`):
- `unprivileged = true`
- `features { nesting = true, keyctl = true, fuse = true }`
- `operating_system { template_file_id = "..." }` — Debian 13 template
- `disk { datastore_id = "<your-encrypted-thinpool>" }`
- `initialization { user_account { keys = [...] }, dns { ... }, ip_config { ... } }` — cloud-init via Proxmox-native fields, plus `user_data_file_id` pointing to a snippet for everything Proxmox's native cloud-init can't express.

Subuid/subgid mapping is the part the module may not handle cleanly. You'll likely need a `lxc_extra_config` block or a post-create local-exec to append the `lxc.idmap` lines to `/etc/pve/lxc/<vmid>.conf`. Worth checking the module's open issues before you start.

### 3. Cloud-init snippet (inside-the-LXC bootstrap)

Stored as a Proxmox snippet, referenced from Tofu. Should:
- Create a non-root service user (e.g. `knowitall`) with a stable UID.
- `loginctl enable-linger knowitall` so the user systemd instance starts at boot without a login session.
- Install: `podman`, `slirp4netns`, `fuse-overlayfs` (fallback if kernel overlayfs misbehaves), `uidmap`, `git`, `curl`.
- Populate `/etc/subuid` and `/etc/subgid` for the service user.
- Clone the knowitall repo (or pull a tagged release) to `/home/knowitall/knowitall`.
- Drop the Quadlet unit at `/home/knowitall/.config/containers/systemd/knowitall.container`.
- Write `/home/knowitall/.config/containers/systemd/knowitall.env` with `KNOWITALL_TOKEN`, `KNOWITALL_OLLAMA_URL`, etc. Don't put the token in the cloud-init snippet directly — pull from a secret backend or inject post-provision.
- `systemctl --user daemon-reload && systemctl --user enable --now knowitall.service` (Quadlet generates the `.service` from the `.container`).

### 4. Quadlet unit

Translated from `deploy/docker-compose.yml`. Use [`podlet`](https://github.com/containers/podlet) to generate a first draft (`podlet compose deploy/docker-compose.yml`), then hand-review. Approximate shape:

```ini
# ~/.config/containers/systemd/knowitall.container
[Unit]
Description=knowitall knowitall server
After=network-online.target
Wants=network-online.target

[Container]
Image=knowitall:latest
ContainerName=knowitall
PublishPort=8765:8765
Volume=%h/knowitall/data:/data:Z
EnvironmentFile=%h/.config/containers/systemd/knowitall.env
HealthCmd=curl -fsS http://127.0.0.1:8765/healthz
HealthInterval=15s
HealthRetries=5
HealthStartPeriod=10s

[Service]
Restart=always

[Install]
WantedBy=default.target
```

Image build: either build locally inside the LXC on first boot (`podman build -f deploy/Dockerfile -t knowitall:latest .`) or push to a registry from CI and `podman pull` here. Local build is simpler for a single-host homelab; CI registry pays off only once you have multiple deploy targets.

### 5. Pulse agent

Install inside the LXC as another Quadlet unit (or as a host package, whichever Pulse documents). Confirms what you noted: the LXC-level view from `mikro` is aggregate; per-container metrics require an in-LXC agent.

## Backups

Stop-then-snapshot via PBS. For a single-container app with file-backed embedded stores (Kùzu WAL, LanceDB), this is the right call:
- Kùzu writes a WAL that needs a clean checkpoint for a consistent snapshot. A running `vzdump` could capture mid-write state.
- Brief downtime (seconds) is acceptable for this workload.
- Schedule during a low-traffic window via a PBS job. Document the expected outage window in your runbook.

If downtime ever becomes unacceptable, the escape hatch is application-level: a periodic `kuzu` export + LanceDB snapshot to a sibling directory, then `vzdump` that without stopping. Cross that bridge when the constraint actually bites.

## Operational notes

- **Updates**: pull the repo, `podman build`, `systemctl --user restart knowitall`. The Quadlet unit regenerates on `daemon-reload` — no `systemctl edit` drift.
- **Logs**: `journalctl --user -u knowitall.service` from inside the LXC. Quadlet pipes container stdout/stderr to journald automatically.
- **Secrets**: `KNOWITALL_TOKEN` belongs in `knowitall.env` (mode 0600, owned by the service user). Don't commit it; inject via your secret tool of choice (or `pass`, or a sealed file restored on first boot).
- **Rollback**: tag releases, keep the previous image (`podman image ls` won't auto-prune). Re-tag previous as `:latest` and restart if a deploy goes bad.

## Known gotchas

Ranked by likelihood of biting:

1. **overlayfs in unprivileged user namespace**. Modern kernels (5.11+) and Proxmox 9.x are fine, but if `podman run` fails with mount errors after a clean `podman pull`, fall back to `fuse-overlayfs` (`storage.conf`: `driver = "overlay"`, `mount_program = "/usr/bin/fuse-overlayfs"`). Install `fuse-overlayfs` in cloud-init pre-emptively.

2. **cgroups v2 delegation**. Resource limits in the Quadlet (`MemoryMax=`, `CPUQuota=`) may silently no-op if the cgroup controllers aren't delegated to the rootless user. Check with `cat /sys/fs/cgroup/user.slice/user-$(id -u).slice/cgroup.controllers`. If `memory` or `cpu` are missing, you need to delegate them in the LXC config or via a drop-in. Not catastrophic — limits just won't apply — but easy to miss.

3. **AppArmor conflict**. Proxmox applies an LXC AppArmor profile; Podman wants its own container profile. Symptom: containers fail to start with permission-denied on things that should work. Workaround is `lxc.apparmor.profile: unconfined` in the LXC config, which trades isolation back. Try without it first.

4. **`/etc/subuid` and `/etc/subgid` math**. The host's subuid mapping for `root` (which the unprivileged LXC uses) must be wide enough to cover the LXC's internal range *and* that range must be wide enough for the rootless user inside to have its own sub-mappings. Two layers of namespace nesting. Get this wrong and `podman` inside the LXC fails with `newuidmap: write to uid_map failed`. The DigitallyRefined guide has working numbers.

5. **systemd-as-PID-1-in-LXC + `systemd --user` interactions**. Mostly works in 2026, but `loginctl enable-linger` is mandatory and journald inside the LXC can behave oddly. If `journalctl --user` shows nothing after a restart, check that `systemd --user` is actually running (`systemctl --user status`).

6. **vzdump snapshot consistency with running Podman**. The reason for stop-then-snapshot above. If you ever try live snapshots, expect occasionally-corrupt backups of the Kùzu WAL.

7. **Kernel module surface**. Anything kernel-side (WireGuard, specific FS modules) must be loaded on the Proxmox host, not in the LXC. knowitall is pure userspace so this doesn't apply today, but flag it if dependencies change.

8. **Backup encryption boundary**. Host-level LUKS protects data at rest on `mikro`. PBS backups go elsewhere — make sure the PBS datastore is also encrypted (PBS supports client-side encryption with a key you hold) or you've broken your own threat model.

## Open questions to resolve before first deploy

- **LUKS unlock at boot**: keyfile on unencrypted partition (unattended reboot works, key is recoverable from physical access) vs. manual passphrase (requires console access after every reboot, but stronger). Pick one and document.
- **Secret injection for `KNOWITALL_TOKEN`**: which mechanism? Manual scp on first boot, sops-encrypted file in the repo, a secret manager? Cloud-init runs once; rotation needs a separate path.
- **Image build location**: in-LXC build (simple, slow rebuilds) vs. CI-pushed registry (faster, more infrastructure). Default to in-LXC until it hurts.
- **Tofu state backend**: local state file is fine for one operator; if you ever want to run Tofu from multiple places, set up a remote backend before the second machine, not after.

## References

- [bpg/terraform-provider-proxmox](https://github.com/bpg/terraform-provider-proxmox) — current OpenTofu provider, Proxmox 9.x compatible
- [kode3tech/terraform-proxmox-lxc](https://github.com/kode3tech/terraform-proxmox-lxc) — LXC module with nesting/keyctl/fuse support
- [woutervanelten/Podman](https://github.com/woutervanelten/Podman) — working rootless Podman in unprivileged Proxmox LXC, with Quadlets
- [DigitallyRefined PVE unprivileged LXC rootless containers guide (Feb 2026)](https://digitallyrefined.github.io/guides/PVE-unprivileged-LXC-rootless-containers) — subuid/subgid math
- [containers/podlet](https://github.com/containers/podlet) — compose → Quadlet converter
- [Red Hat: Make systemd better for Podman with Quadlet](https://www.redhat.com/en/blog/quadlet-podman)
