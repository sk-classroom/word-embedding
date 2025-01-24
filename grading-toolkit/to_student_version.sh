if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <teacher_notebook.ipynb> <student_notebook.ipynb> <encryption_password>"
    exit 1
fi

teacher_notebook=$1
student_notebook=$2
encryption_password=$3
tmp_notebook="assignment.ipynb"

teacher_notebook_enc=${teacher_notebook%.ipynb}.ipynb.enc

openssl enc -aes256 -in $teacher_notebook -out $teacher_notebook_enc -pass pass:$encryption_password

cp $teacher_notebook $tmp_notebook

jupyter nbconvert --to notebook --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["answer"]' $teacher_notebook --inplace

# Then use jq to rename "answer" to "gradable"
jupyter nbconvert --to notebook --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags='["gradable"]' $tmp_notebook --inplace
jq '.cells[].metadata.tags |= if . != null then map(if . == "answer" then "gradable" else . end) else . end' $tmp_notebook -i


mv $teacher_notebook $student_notebook

