# 📝 Git + GitHub Cheatsheet

## Setup

```
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

## Start a Repo

```
git init                  # new repo
git clone URL             # copy existing repo
```

## Basic Workflow

```
git status                # check changes
git add file.txt          # stage file
git add .                 # stage all
git commit -m "Message"   # commit
git push origin main      # push to GitHub
git pull origin main      # update from GitHub
```

## Branching

```
git branch new-feature    # create branch
git checkout new-feature  # switch
git checkout -b new-feature # create + switch
git merge new-feature     # merge into current branch
git branch -d new-feature # delete branch
```

## Logs & History

```
git log                   # view history
git diff                  # show changes
git diff --staged         # show staged changes
```

## Undo and Fix

```
git reset file.txt        # unstage file
git reset --soft HEAD~1   # undo last commit, keep changes
git reset --hard HEAD~1   # undo last commit, discard changes
git revert <hash>         # revert safely
```

## Extras

```
git stash                 # save work without commit
git cherry-pick <hash>    # copy a commit
git remote -v             # show remotes
```

## ✅ Key Mental Model to Remember

* Working Dir → Staging Area → Repo → GitHub
* Always add → commit → push
* Branches are just labels to commits.
