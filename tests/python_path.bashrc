PRIMUS_PATH=`pwd`

setup_pythonpath() {
    # Get site-packages directory for current Python environment
    local site_packages
    site_packages=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

    local third_party_path="${PRIMUS_PATH}/third_party"
    local third_party_pythonpath=""

    # Define backend names that can be overridden via environment variables
    local CUSTOM_BACKENDS=("megatron" "torchtitan")
    declare -A CUSTOM_BACKEND_PATHS

    # Load backend paths from environment variables (e.g., MEGATRON_PATH)
    for backend in "${CUSTOM_BACKENDS[@]}"; do
        # Convert backend name to uppercase and append _PATH (e.g., MEGATRON_PATH)
        env_var_name="$(echo "${backend}_path" | tr '[:lower:]' '[:upper:]')"
        backend_path="${!env_var_name}"
        if [[ -n "$backend_path" ]]; then
            check_dir_nonempty "$env_var_name" "$backend_path"
            CUSTOM_BACKEND_PATHS["$backend"]="$backend_path"
        fi
    done

    declare -A DIR_TO_BACKEND=(
        ["Megatron-LM"]="megatron"
        ["torchtitan"]="torchtitan"
    )
    # Collect third_party paths, excluding overridden backends
    while IFS= read -r dir; do
        base_name=$(basename "$dir")
        base_name="${DIR_TO_BACKEND[$base_name]}"
        if [[ -n "${CUSTOM_BACKEND_PATHS[$base_name]}" ]]; then
            continue
        fi
        third_party_pythonpath+="${dir}:"
    done < <(find "${third_party_path}" -mindepth 1 -maxdepth 1 -type d -exec realpath {} \;)

    third_party_pythonpath="${third_party_pythonpath%:}"  # Remove trailing colon

    # Start building final PYTHONPATH
    local full_pythonpath="${site_packages}:${PRIMUS_PATH}:${third_party_pythonpath}"

    # Prepend custom backend paths if defined
    for backend in "${CUSTOM_BACKENDS[@]}"; do
        custom_path="${CUSTOM_BACKEND_PATHS[$backend]}"
        [[ -n "$custom_path" ]] && full_pythonpath="${custom_path}:${full_pythonpath}"
    done

    export PYTHONPATH="${full_pythonpath}:${PYTHONPATH}"
    echo "[INFO] PYTHONPATH is set to: ${PYTHONPATH}"
}

setup_pythonpath
