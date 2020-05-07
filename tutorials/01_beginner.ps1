# simple script to run the python file and echo the output to both shell and log file
$file = ".\tutorials\01_beginner"
py "${file}.py" | Tee-Object -FilePath "${file}.log"