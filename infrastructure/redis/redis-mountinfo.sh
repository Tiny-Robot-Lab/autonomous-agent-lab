redis_mountinfo_is_tmpfs() {
    mountinfo_path=$1
    expected_mountpoint=$2

    [ -r "$mountinfo_path" ] || return 1
    awk -v expected_mountpoint="$expected_mountpoint" '
        {
            filesystem = ""
            for (field = 6; field <= NF; field++) {
                if ($field == "-") {
                    filesystem = $(field + 1)
                    break
                }
            }
            row_count++
            devices[row_count] = $3
            roots[row_count] = $4
            filesystems[row_count] = filesystem
            if ($5 == expected_mountpoint) {
                target_entries++
                if ($4 == "/" && filesystem == "tmpfs") {
                    target_device = $3
                    found = 1
                }
            }
        }
        END {
            for (row = 1; row <= row_count; row++) {
                if (devices[row] == target_device && roots[row] == "/" && filesystems[row] == "tmpfs") {
                    matching_superblocks++
                }
            }
            if (target_entries != 1 || !found || matching_superblocks != 1) {
                exit 1
            }
        }
    ' "$mountinfo_path"
}
