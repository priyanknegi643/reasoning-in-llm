Any additions to the repository, are to be added here. 

-----------Additions -----------

1. Added the nextlat repo as a github sumbodule.

To add it ( since it won't be there in any system in the form of files ) 

- git submodule update --init --recursive

2. Uv environment has been initialized.

[Remember] always initialize venv by ->
 source .venv/bin/activate  ( in the pod )
 deactivate 

- for any new repo, which has ->
 If the project depends on the submodule -> ( command to be executed from root ) 
- uv add <path/to/file>  or uv add -r path/to/requirements.txt

 If the repo is a python package of it's own ( it contains it's own pyproject.toml file )
- uv add <path/to/folder of submodule>

    
