while true; do 
    read answer < /tmp/traccc_pr_queue;
    if [[ $answer == *"P"* ]]; then
        echo "Running physics check on $answer"
        uv run python check_physics.py $(echo $answer | sed -E 's/^([[:digit:]]+).*$/\1/g' --);
    else
        echo "Skipping physics check on $answer"
    fi
    if [[ $answer == *"C"* ]]; then
        echo "Running compute check on $answer"
        uv run python check_compute.py $(echo $answer | sed -E 's/^([[:digit:]]+).*$/\1/g' --);
    else
        echo "Skipping compute check on $answer"
    fi
done
