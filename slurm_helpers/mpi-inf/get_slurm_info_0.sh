#!/bin/bash                                                                                   
                                                                                              
#echo "========== Nodes with PARTIALLY USED GPUs =========="                                  
#printf "%-15s %-7s %-7s %-7s\n" "Node" "Total" "Used" "Free"                                 
                                                                                              
# Counters                                                                                    
partial_count=0                                                                               
free_count=0                                                                                  
total_partial_free_gpus=0                                                                     
total_completely_free_gpus=0                                                                  
declare -a free_nodes=()                                                                      
                                                                                              
# Free GPU bucket counters                                                                    
count_free_1=0                                                                                
count_free_2=0                                                                                
count_free_3=0                                                                                
count_free_4=0                                                                                
                                                                                              
# Function to count GPUs in IDX string                                                        
count_gpus_in_idx() {                                                                         
    local idx_str=$1                                                                          
    local count=0                                                                             
    IFS=',' read -ra parts <<< "$idx_str"                                                     
    for p in "${parts[@]}"; do                                                                
        if [[ "$p" =~ ^[0-9]+$ ]]; then                                                       
            count=$((count + 1))                                                              
        elif [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]]; then                                          
            start=$(echo "$p" | cut -d'-' -f1)                                                
            end=$(echo "$p" | cut -d'-' -f2)                                                  
            count=$((count + end - start + 1))                                                
        fi                                                                                    
    done                                                                                      
    echo $count                                                                               
}                                                                                             
                                                                                              
# Main loop                                                                                   
while read -r node gres used; do                                                              
    total=$(echo "$gres" | grep -oP "gpu(:[a-zA-Z0-9_]+)?:\K[0-9]+")                          
    [[ -z "$total" ]] && total=0                                                              
                                                                                              
    idx_range=$(echo "$used" | grep -oP "IDX:\K[^)]*")                                        
    if [[ -n "$idx_range" ]]; then                                                            
        used_gpus=$(count_gpus_in_idx "$idx_range")                                           
    else                                                                                      
        used_gpus=0                                                                           
    fi                                                                                        
                                                                                              
    free=$((total - used_gpus))                                                               
    (( free < 0 )) && free=0                                                                  
                                                                                              
    if (( used_gpus > 0 && free > 0 )); then                                                  
        #printf "%-15s %-7s %-7s %-7s\n" "$node" "$total" "$used_gpus" "$free"                
        partial_count=$((partial_count + 1))                                                  
        total_partial_free_gpus=$((total_partial_free_gpus + free))                           
                                                                                              
        # Bucket count                                                                        
        case $free in                                                                         
            1) count_free_1=$((count_free_1 + 1));;                                           
            2) count_free_2=$((count_free_2 + 1));;                                           
            3) count_free_3=$((count_free_3 + 1));;                                           
            4) count_free_4=$((count_free_4 + 1));;                                           
        esac                                                                                  
                                                                                              
    elif (( used_gpus == 0 && total > 0 )); then                                              
        free_nodes+=("$node:$total")                                                          
        free_count=$((free_count + 1))                                                        
        total_completely_free_gpus=$((total_completely_free_gpus + total))                    
                                                                                              
        # Bucket count                                                                        
        case $total in                                                                        
            1) count_free_1=$((count_free_1 + 1));;                                           
            2) count_free_2=$((count_free_2 + 1));;                                           
            3) count_free_3=$((count_free_3 + 1));;                                           
            4) count_free_4=$((count_free_4 + 1));;                                           
        esac                                                                                  
    fi                                                                                        
done < <(sinfo -p gpu17 --noheader --Format=NodeHost,Gres,GresUsed)                             
                                                                                              
echo ""                                                                                       
echo "========== Nodes with COMPLETELY FREE GPUs =========="                                  
printf "%-15s Free GPUs\n"                                                                    
for entry in "${free_nodes[@]}"; do                                                           
    IFS=":" read -r node free <<< "$entry"                                                    
    printf "%-15s %s\n" "$node" "$free"                                                       
done                                                                                          
                                                                                              
echo ""                                                                                       
echo "==================== Summary ===================="                                      
echo "Partially used nodes:       $partial_count"                                             
echo "?~F~R Free GPUs on partials:    $total_partial_free_gpus"                               
echo "Completely free nodes:      $free_count"                                                
echo "?~F~R Free GPUs on free nodes:  $total_completely_free_gpus"                            
echo "-------------------------------------------------"                                      
echo "TOTAL free GPUs:            $((total_partial_free_gpus + total_completely_free_gpus))"
                                                                                              
echo ""                                                                                       
echo "========== Free GPU Count Breakdown =========="                                         
echo "Nodes with 1 GPU free:      $count_free_1"                                              
echo "Nodes with 2 GPUs free:     $count_free_2"                                              
echo "Nodes with 3 GPUs free:     $count_free_3"                                              
echo "Nodes with 4 GPUs free:     $count_free_4"
