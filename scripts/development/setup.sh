#!/bin/bash

# This script sets up automatically the upstream and origin remotes in a git cloned repo.  
# Tries ssh and https (if ssh fails) connections by asking your host username.
# Run this script once after cloning a repo.
# Upstream remote -> original project (current origin when cloning)
# Origin remote -> your fork (must be created beforehand)
echo "Set up automatically the upstream and origin remote repos" 

# Start part to modify
git_group=neutrons
git_project=PyRS
git_host=github.com
# End part to modify

# Get git host username from your account
read -p "ENTER ${git_host} username: " git_username
if test -z "${git_username}"
then
  echo "${git_host} username can't be empty"
  exit
fi

# Configure origin and upstream repo via SSH (preference) or HTTPS if SSH fails
# SSH users make sure your system public key is registed in gitlab user settings > SSH Keys 
# (e.g. ~/.ssh/id_rsa.pub)
remote_origin=""
remote_upstream=""

# First try connecting via ssh
echo "Checking SSH access to host " ${git_host} " for user " ${git_username}
ssh -o ConnectTimeout=10 -T git@${git_host} 2>/dev/null

if [ $? -eq 1 ]
then
  echo "SSH success."
  remote_origin=git@${git_host}:${git_username}/${git_project}.git
  remote_upstream=git@${git_host}:${git_group}/${git_project}.git
else
  echo "SSH failed. Using HTTPS."
  remote_origin=https://${git_host}/${git_username}/${git_project}.git
  remote_upstream=https://${git_host}/${git_group}/${git_project}.git
fi

echo ""
echo "Setting upstream and origin (user fork) remotes"
echo "origin: " ${remote_origin}
echo "upstream: " ${remote_upstream}
echo ""

git remote set-url origin ${remote_origin}
# Removes any existing upstream remote
git remote rm upstream
git remote add upstream ${remote_upstream}
# fetch all remote (upstream and origin) branches
git fetch --all -p

echo ""
echo "Local master branch points to upstream/master, NEVER push to it"
git checkout master
git branch --set-upstream-to=upstream/master master
# only fast forward to "protect" master branch
git config --add branch.master.mergeOptions --ff-only

# log message uses default 20 one-liner commit messages
git config merge.log 20

echo ""
echo "Success, verify upstream and origin: "
git remote -v

exit 0
