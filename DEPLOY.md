# Deploying the Orchestrated Pipeline (Plan 2)

Prerequisites: Nebius account + project, GPU presets chosen for prep/synth, k3s
access (`k8s/kubeconfig`), R2 + Gemini + HF creds, the buzz-bot Neon `DATABASE_URL`.

## 1. Build & import images
```bash
# CPU image (orchestrator + cpu-text + cpu-mux)
docker build -f Dockerfile.cpu -t dub-orch-cpu:latest .
# GPU image (prep + synth) — pushed to the registry Nebius pulls from
docker build -f Dockerfile.gpu -t <registry>/dub-gpu:latest .
docker push <registry>/dub-gpu:latest
```
Import the CPU image into k3s containerd (per buzz-bot's image-import convention),
since the Deployments use `imagePullPolicy: IfNotPresent`.

## 2. Database
Apply the orchestrator schema to Neon, and the buzz-bot transcript table:
```bash
psql "$DATABASE_URL" -f src/orchestrator/schema.sql
psql "$BUZZBOT_DATABASE_URL" -f ../buzz-bot/migrations/021_transcript_jobs.sql
```
(The orchestrator also calls `store.init_schema()` on startup as a safety net.)

## 3. Secrets + manifests
```bash
cp k8s/secret.example.yaml k8s/secret.yaml   # fill in real values (gitignored)
kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/orchestrator.yaml -f k8s/cpu-text.yaml -f k8s/cpu-mux.yaml
kubectl -n buzz-bot rollout status deploy/dub-orchestrator
```
Add `ORCH_BASE_URL=https://orch.buzz-bot.top` and `ORCH_DISPATCH_SECRET=<same as
ORCH_CALLBACK_SECRET>` to the buzz-bot secret, then redeploy buzz-bot (flag still off).

## 4. Nebius smoke test (paid — run once)
Dispatch a short clip and watch one GPU job complete:
```bash
curl -X POST https://orch.buzz-bot.top/dispatch \
  -H "X-Dispatch-Token: $ORCH_CALLBACK_SECRET" -H "Content-Type: application/json" \
  -d '{"run_id":"smoke1","workflow_type":"transcribe","episode_id":<short-ep>,
       "audio_url":"<short-clip-url>","callback_url":"https://app.buzz-bot.top/internal/transcript_result"}'
# verify: a Nebius gpu-prep job runs, /callback advances the run, transcript appears.
```

## 5. Cut over + verify
```bash
# In Telegram, as an admin:
/flag dub_orchestrator on
```
- Trigger a dub from the Mini App → confirm it runs via the orchestrator (Nebius
  prep → cpu-text → Nebius synth → cpu-mux) and the episode plays dubbed.
- Open an un-transcribed episode's subtitle panel → tap **Generate transcript** →
  confirm cues appear.
- Roll back instantly if needed: `/flag dub_orchestrator off` (reverts to RunPod).

## 6. Decommission (later)
Once the flag has been on in production without issue, remove `src/worker.py` and
the RunPod path in a follow-up cleanup.
