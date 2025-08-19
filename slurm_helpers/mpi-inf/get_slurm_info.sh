#!/bin/bash
# sinfo -p gpu --Format=NodeHost,Gres,GresUsed
# List of GPU partitions to check
PARTITIONS=(gpu16 gpu17 gpu20 gpu22 gpu24)

echo "================ GPU Cluster Summary ================"

for PARTITION in "${PARTITIONS[@]}"; do
    total_partition_gpus=0
    total_partition_allocated=0
    total_partition_free=0

    # Get list of nodes in this partition
    nodes=$(sinfo -p $PARTITION -N -h -o "%N")

    echo ""
    echo "Partition: $PARTITION"

    for node in $nodes; do
        # Extract total GPUs from the Gres line (case-insensitive match)
        total_gpus=$(scontrol show node $node | grep -oiP "Gres=gpu(:[a-zA-Z0-9_]+)?:\K[0-9]+" | head -n1)
        total_gpus=${total_gpus:-0}

        # Extract requested GPUs from squeue %b (covers all valid GRES types)
        allocated=$(squeue -w $node -h -o "%b" | grep -oiP "gpu(:[a-zA-Z0-9_]+)?:\K[0-9]+" | awk '{sum+=$1} END{print sum+0}')

        # Fallback: assume 1 GPU per job if %b is empty or unparseable
        if [[ "$allocated" -eq 0 ]]; then
            job_count=$(squeue -w $node -h -o "%i" | wc -l)
            allocated=$job_count
        fi

        # Calculate free GPUs
        free_gpus=$((total_gpus - allocated))
        (( free_gpus < 0 )) && free_gpus=0

        # Update partition totals
        total_partition_gpus=$((total_partition_gpus + total_gpus))
        total_partition_allocated=$((total_partition_allocated + allocated))
        total_partition_free=$((total_partition_free + free_gpus))

        echo "  $node: total=$total_gpus, allocated=$allocated, free=$free_gpus"
    done

    echo "→ Partition Summary: total=$total_partition_gpus, allocated=$total_partition_allocated, free=$total_partition_free"
done

echo ""
echo "======================================================"

