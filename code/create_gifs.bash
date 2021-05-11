#!/usr/bin/env bash
# Author: "abpwrs"
# Date: 20210511

# args: NONE -- hardcoded

ls /Shared/sinapse/cjohnson/all_inferrrence_removed_bad/*.nii.gz | time parallel --eta --jobs 24 python image_to_gif.py --image {} --axis l
ls /Shared/sinapse/cjohnson/all_inferrrence_removed_bad/*.nii.gz | time parallel --eta --jobs 24 python image_to_gif.py --image {} --axis p
ls /Shared/sinapse/cjohnson/all_inferrrence_removed_bad/*.nii.gz | time parallel --eta --jobs 24 python image_to_gif.py --image {} --axis s


