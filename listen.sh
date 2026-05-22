PIPE_NAME=/tmp/traccc_pr_queue

while true; do
    if [ ! -p ${PIPE_NAME} ]; then
        echo "Pipe doesn't exist, making it."
        mkfifo ${PIPE_NAME}
        chmod 777 ${PIPE_NAME}
        sleep 60
    fi

    set -- $(stat --format '%U %a' ${PIPE_NAME})

    if [[ $1 != "sswatman" ]]; then
        echo "File ${PIPE_NAME} is not owned by sswatman"
        sleep 60
        continue
    fi
    if [[ $2 != 777 ]]; then
        echo "File ${PIPE_NAME} does not have permissions 777"
        sleep 60
        continue
    fi

    read answer < ${PIPE_NAME};
    if [[ $answer == *"P"* ]]; then
        echo "Running physics check on $answer"
        PYTHONPATH= uv run python check_physics.py $(echo $answer | sed -E 's/^([[:digit:]]+).*$/\1/g' --);
    else
        echo "Skipping physics check on $answer"
    fi
    if [[ $answer == *"C"* ]]; then
        echo "Running compute check on $answer"
        PYTHONPATH= uv run python check_compute.py $(echo $answer | sed -E 's/^([[:digit:]]+).*$/\1/g' --);
    else
        echo "Skipping compute check on $answer"
    fi
done
