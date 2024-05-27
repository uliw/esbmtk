#!/bin/bash
# Tangle all code from the org files and move it to the 
# examples directory, # convert org mode files to rst
# this requires that a suitable emacs instance with the
# necessaryt libraries is already already running

cd /home/uliw/user/python-scripts/esbmtk/docs/manual/
for i in *.org; do
    [ -f "$i" ] || break
    emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"$i\")) (end-of-buffer) (org-babel-tangle) (kill-buffer))"
   
     emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"$i\")) (end-of-buffer) (org-rst-export-to-rst) (kill-buffer))"
done

cp *.py /home/uliw/user/python-scripts/esbmtk_examples/Examples_from_the_manual/

cd /home/uliw/user/python-scripts/esbmtk/
emacsclient -e "(progn (switch-to-buffer (find-file-noselect \"CHANGELOG.org\")) (end-of-buffer) (org-rst-export-to-rst) (kill-buffer))"
pandoc README.org -o README.md