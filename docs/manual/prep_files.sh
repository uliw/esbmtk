#!/bin/bash
# Tangle all code from the org files and move it to the 
# test & examples directory, # convert org mode files to rst
# this requires that a suitable emacs instance with the
# necessaryt libraries is already already running

cd /home/uliw/user/python-scripts/esbmtk/docs/manual/
for i in *.org; do
    [ -f "$i" ] || break
    emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"$i\")) (end-of-buffer) (org-babel-tangle) (kill-buffer))"
   
     emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"$i\")) (end-of-buffer) (org-rst-export-to-rst) (kill-buffer))"
done

mv *test*.py /home/uliw/user/python-scripts/esbmtk/tests/

# regenerate images
directory="/home/uliw/user/python-scripts/esbmtk/docs/manual/"

# Loop through all Python files in the directory and execute them
for script in "$directory"/*.py; do
  # Check if the file exists (in case there are no .py files)
  if [ -f "$script" ]; then
    echo "Executing $script..."
    python "$script"  # You can replace 'python' with 'python3' if needed
  else
    echo "No Python scripts found in $directory"
  fi
done

# move python files to Example repository
mv *.py /home/uliw/user/python-scripts/esbmtk_examples/Examples_from_the_manual/

cd /home/uliw/user/python-scripts/esbmtk/
emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"CHANGELOG.org\")) (end-of-buffer) (org-rst-export-to-rst) (kill-buffer))"
pandoc README.org -o README.md

emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"README.org\")) (end-of-buffer) (org-gfm-export-to-markdown) (kill-buffer))"
