# Day 27/6
## BASIC THEORY

1. Git is a distributed version control system
   - Tracks changes in source code over time
   - Works locally without internet connection

2. File States
   - Untracked: new file, Git does not know about it
   - Tracked: file already added to Git
     - Unmodified: no changes since last commit
     - Modified: changed but not staged
     - Staged: ready to be committed

3. Three Areas
   - Working Directory: where you edit files
   - Staging Area: where you prepare changes for commit
   - Repository: where commits are permanently stored

4. Basic Workflow
   - Edit file → git add → git commit → git push

5. Common Command GIT:
   - git add:
     - git add filename : stage a specific file
     - git add .
   - git commit
     - git commit -m "message" : commit with message
     - git commit -a -m "message" : skip staging, commit all tracked files
     - git commit --amend : edit the last commit
   - git push & pull
     - git push origin branch : upload to remote
     - git pull : fetch and merge from remote

9. Branch
   - git branch : list branches
   - git checkout name : switch branch
   - git checkout -b name : create and switch

10. Stash:
    - git stash -u : create temporary memories for all changes not be committed before checkout other branch
      - => and then back this branch, use `git stash pop` 
    - git stash list: see all saved stashes
    - git stash show: see what changed in lastest stash
    - git stash apply stash@{n} : choose stash from list
    - git stash pop : apply the lasted stash and remove it in stash

11. log & Diff:
    - git log --oneline: display detail of each commit in one line
    - git diff : See the changes in local , not yet use `git add`
    - git diff --staged or --cached : See the changes after use git add but not commited.
    - git diff <branch-A> <branch-B>: See differents between 2 branchs
    - git diff <commit1> <commit2> : Compare two commits
    
    ### Combo can use before push code to Github
    - git status -> git diff -> git add -> git commit -> git log -> git push

12. Merge:
    - git merge <branch-start>: realize merge to <branch-end> form <branch-start>
    - process: 
      - git checkout develop : move to <branch-end>
      - git pull origin develop: pull nearly code from github 
      - git merge feature/week1 
    - Need to push to Github and choose Merge pull request
    
    #### Solve conflict :
    - In process merging, if wanna undo command merge: `git merge --abort` 

13. Reset & Revert:
    - git reset HEAD~1 : back to before nearly commit ( cancel the last commit)
      - ** like A--B--C => A--B ( commit error C was deleted)
    - git reset --hard : Force to reset ( Unsafety) only use in *LOCAL*
    - git revert: Create new commit to revert the changes of the commit error
      - ** like A--B--C--D (D is new commit for error commit C )
    - => `git revert` is more safety than `git reset` when collab with others

14. Fetch:
    - git fetch origin : download all history committed, the branchs to local and update remote-tracking branches ( origin/main, origin/develop,..)
    - `origin`: can be understood that directory to store project's internet address

### Noti: 
1. Always check `where am i stand?` before create new branch
2. Should checkout -b from branch main or develop ( not feature/), follow these steps:
   - git checkout develop
   - git pull origin develop
   - git checkout -b feature/X