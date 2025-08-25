## Guide: How to mirror your fork into a new private repo?

### 1. Create an empty private repo on GitHub (UI or CLI):

`gh repo create <private-name> --private --confirm   # or use the UI`


### 2. From your local clone of the fork (where origin is your fork, upstream is the original):

```
# Push EVERYTHING (all branches, tags, refs) to the new private repo
git remote add private git@github.com:<you>/<private-name>.git
git push --mirror private
```


### 3. (Optional) Make the private repo your new default origin:

```
git remote rename origin public          # keep your old fork remote as 'public' for PRs
git remote rename private origin         # 'origin' now points to the private repo
git remote -v
```

### 4. Keep syncing from the public upstream:

```
git remote add upstream https://github.com/<upstream-owner>/<repo>.git  # if not already
git fetch upstream
# later to sync:
git checkout main
git pull --rebase upstream main
git push origin main
```


### 5. Opening PRs upstream? Push just the PR branch to the public fork when needed:

```
git push public my-feature:my-feature
# open PR from <you>/<public-fork>:my-feature -> upstream:main
```


You can delete the public fork later if you don’t need PRs from it—but keep it if you plan to contribute upstream.

### Tiny Cheat-Sheet
See remotes: `git remote -v`

Add remote: `git remote add <name> <url>`

Rename remote: `git remote rename <old> <new>`

Change URL: `git remote set-url <name> <url>`

Remove remote: `git remote remove <name>`

Fetch: `git fetch <remote>`

Pull with rebase: `git pull --rebase <remote> <branch>`

Push: `git push <remote> <branch>`

Track: `git push -u <remote> <branch>`

Mirror everything (careful!): `git push --mirror <remote>`
