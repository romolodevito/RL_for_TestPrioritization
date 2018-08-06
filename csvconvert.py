import csv
import json
import datetime
import pandas as pd


jdata=[]

## Opening the JSON dump file 
jfile = open('run_183.json')

## The CSV file where the data has to be written after formatting and hashing
final_file = open('siemens_data_180.csv','w')

for line in jfile:
	jdata.append(json.loads(line))

zero_test_builds = 0 

## Checking for builds which have zero test results
## We remove the test builds having zero test results, otherwise it gives errors whem RETECS is applied
for i in range(0,len(jdata)):
	i=i-zero_test_builds
	if(len(jdata[i]['result'])==0):
		zero_test_builds+=1
		jdata.remove(jdata[i])

##Printing the No of Zero Builds
print(zero_test_builds)

index=1
x=0
nfailed=0
npassed=0

## Anonymizing the test case names and giving ids to them. idd can start from any number
idd=10000


## We need only certain fields from the Dump filea and that too in a certain format. Dictionaries are created so that the test history and the last run and the idd can be mapped to the test name.
priority=dict()
name_convert =dict()
last_results=dict()
lastrun = dict()

### Assign Priority is the dictionary which can be used to assign priority to a test case . It can be used in further datasets where the priority of the test case is also given.
#assign_priority=dict()



### For testing the effect of priority, priorities were assigned to the different test types
# num_prio=0
# pri=[5,15,25,35,60,80,100,120]

for i in range(0,len(jdata)):
	for j in range(0,len(jdata[i]["result"])):
		## Converting the date to the required format
		jdata[i]["result"][j]["startTime"] = jdata[i]["result"][j]["startTime"].replace("T"," ")
		jdata[i]["result"][j]["startTime"]=jdata[i]["result"][j]["startTime"][:-5]
		jdata[i]["result"][j]["endTime"]=jdata[i]["result"][j]["endTime"][:-5]
		jdata[i]["result"][j]["endTime"] = jdata[i]["result"][j]["endTime"].replace("T"," ")

		## Giving Ids to the test cases which will act as the primary key afterwards (Anonymizing the data)
		if(jdata[i]["result"][j]["fullyQualifiedName"] not in name_convert):
			name_convert[jdata[i]["result"][j]["fullyQualifiedName"]]=idd
			last_results[idd]=[]
			idd+=1		



final_file.write('Id;Name;Duration;CalcPrio;LastRun;LastResults;Verdict;Cycle;Priority\n')
cycle=1


## Last Run for every test case is initialized with the same value.
for i in range(len(jdata)-1,0,-1):
	for j in range(0,len(jdata[i]["result"])):
		idd=name_convert[jdata[i]["result"][j]["fullyQualifiedName"]]
		if(idd not in lastrun):
			lastrun[idd]=jdata[len(jdata)-1]["result"][0]["endTime"]


### Assigning the Priority depending upon the type of the test case.
### Used for Testing the effect of Priority

# for i in range(0,len(jdata)):
# 	for j in range(0,len(jdata[i]["result"])):
# 		idd=name_convert[jdata[i]["result"][j]["fullyQualifiedName"]]
# 		if(jdata[i]["result"][j]["testType"] not in priority):
# 			priority[jdata[i]["result"][j]["testType"]]=pri[num_prio]
# 			num_prio+=1

# 		if(idd not in assign_priority):
# 			assign_priority[idd]=priority[jdata[i]["result"][j]["testType"]]


## Count Stores the no of test cases having abnormality in the starting and the ending times.
count=0

for i in range (len(jdata)-2,-1,-1):
	### THe data from 119 to 123 build in the JSON File given was out of order .So ignored that data .
	##The below two lines should be deleted after getting the data from the Server in a proper way .
	if(((len(jdata)-i)>=119 and (len(jdata)-i)<=123)):
		continue

	for j in range(0,len(jdata[i]["result"])):
		idd=name_convert[jdata[i]["result"][j]["fullyQualifiedName"]]
	
		## Either we don't consider these test cases in the subsequent runs or we ignore them for this particular run only
		## IN the below code, if the endTime of a particular test is 0001-01-01 00:00:00,we ignore it , otherwise it will introduce an error into the RETECS code.
		if(str(jdata[i]["result"][j]["endTime"]) == "0001-01-01 00:00:00" ):
			count+=1
			continue

		else:	
			if(jdata[i]["result"][j]["status"] == "Passed"):
				x=0
				npassed +=1
			else:
				x=1
				nfailed+=1
			final_file.write(str(index) +';'+ str(idd)+';'+ str(jdata[i]["result"][j]["duration"]) +';'+ '0' + ';'+ str(lastrun[idd]) + ';'+str(last_results[idd]) +';' +str(x)+';'+ str(cycle))
			lastrun[idd]=jdata[i]["result"][j]["endTime"]
			last_results[idd].append(x)
			if(not(i==len(jdata) and j==len(jdata[i]["result"]))):
				final_file.write('\n')
			index+=1
	cycle+=1
	
# print(t_failed)

## Count indicates the no of test cases having a discrepancy in the start and end time and thus, were ignored.
print(count)
print(nfailed)
print(npassed)