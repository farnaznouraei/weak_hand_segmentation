#!/bin/sh

<< 'MULTILINE-COMMENT'

python run.py --model default --first ./egohands_0.jpg --second ./egohands_1.jpg --out ./out0.flo
python run.py --model default --first ./egohands_1.jpg --second ./egohands_2.jpg --out ./out1.flo
python run.py --model default --first ./egohands_2.jpg --second ./egohands_3.jpg --out ./out2.flo
python run.py --model default --first ./egohands_3.jpg --second ./egohands_4.jpg --out ./out3.flo
python run.py --model default --first ./egohands_4.jpg --second ./egohands_5.jpg --out ./out4.flo
python run.py --model default --first ./egohands_5.jpg --second ./egohands_6.jpg --out ./out5.flo
python run.py --model default --first ./egohands_6.jpg --second ./egohands_7.jpg --out ./out6.flo
python run.py --model default --first ./egohands_7.jpg --second ./egohands_8.jpg --out ./out7.flo
python run.py --model default --first ./egohands_8.jpg --second ./egohands_9.jpg --out ./out8.flo
python run.py --model default --first ./egohands_9.jpg --second ./egohands_10.jpg --out ./out9.flo
python run.py --model default --first ./egohands_10.jpg --second ./egohands_11.jpg --out ./out10.flo
python run.py --model default --first ./egohands_11.jpg --second ./egohands_12.jpg --out ./out11.flo
python run.py --model default --first ./egohands_12.jpg --second ./egohands_13.jpg --out ./out12.flo
python run.py --model default --first ./egohands_13.jpg --second ./egohands_14.jpg --out ./out13.flo
python run.py --model default --first ./egohands_14.jpg --second ./egohands_15.jpg --out ./out14.flo
MULTILINE-COMMENT


a=0

while [ $a -lt 10 ]
do
   echo $a
   if [ $a -eq 25 ]
   then
      break
   fi
    i= $a
    a=` expr $a + 1`
    python run.py --model default --first "./egohands_$i.jpg" --second "./egohands_$a.jpg" --out "./out$i.flo"
done
