1) numbering "m"odules for sorting: [src/m01_<name>.py, src/m02_<name>.py, ] (start with letter "m" to not get into import error with number as prefix)
-- the number m*[01, 02, 3.., 0n] must NOT be REPEATED
2) whem moving form current/previous phase to next phase, move the current python files to @src/legacy/ directory and rename the existing files as per nature of next phase
3) create common / UTILS files here @src/utils
4) note:for both (inference and training) on Nvidia's GPU only. no M1/CPU fallback
keep M1 macbook for only CPU or API based operations
5) in each python script, keep Docstring limited to terminal commands (covering all arguments) to be executed starting max 2 lines of explanation about the code
5.1) each command must have  `python -u src/*.py --args arg_name 2>&1 | tee logs/<log_name>log` format
6) TEST [`py_compile` && `--help` && `ast`] using VIRTUAL ENVIRONMENT 
`source venv_Agnt_Algnmt/bin/activate && python -m py_compile src/m0*.py && echo "All syntax checks passed"`
`source venv_Agnt_Algnmt/bin/activate && python src/m0*.py --help`
