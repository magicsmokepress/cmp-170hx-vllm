#!/bin/bash
# $1=gpu index  $2=url  $3=model  $4=default_watts  $5=conc  $6=ntok  $7=caps...
GPU=$1; URL=$2; MODEL=$3; DEF=$4; CONC=$5; NTOK=$6; shift 6; CAPS="$@"
restore(){ sudo -n nvidia-smi -i $GPU -pl $DEF >/dev/null 2>&1; echo "[restored GPU$GPU to ${DEF}W]" >&2; }
trap restore EXIT
for cap in $CAPS; do
  sudo -n nvidia-smi -i $GPU -pl $cap >/dev/null 2>&1 || { echo "cap $cap rejected" >&2; continue; }
  sleep 2
  for rep in 1 2; do
    python3 bench.py --url "$URL" --model "$MODEL" --gpu $GPU --key "$VLLM_API_KEY" \
      --ntok $NTOK --conc $CONC --label "gpu${GPU}_cap${cap}_c${CONC}_r${rep}"
  done
done
