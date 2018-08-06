import requests
import json
import datetime

## Anonymize the test case by assigning an ID to it.
## Initialize the lastrun and the Last Results field for the new test cases.
def convert_name_to_idd(jdata):
	global idd
	global name_convert
	global lastrun 
	global initial_timestamp
	global last_results
	# print(len(jdata[0]["suites"]))
	for i in range(len(jdata[0]["suites"])):
		for j in range(len(jdata[0]["suites"][i]["cases"])):
		## Giving Ids to the test cases which will act as the primary key afterwards (Anonymizing the data)
		## Here the Class Name Combined with the Name is considered th basis for distinguishing a test case
			full_name= jdata[0]["suites"][i]["cases"][j]["className"] + jdata[0]["suites"][i]["cases"][j]["name"]
			if(full_name not in name_convert):
				name_convert[full_name]=idd
				## Initializing the last results field
				last_results[idd]=[]
				## Initializing the lastrun time
				lastrun[idd]=initial_timestamp
				idd+=1

## Writing into the CSV File
def write_to_csv(jdata):
	global cycle
	global serial_no
	global last_results 
	global lastrun
	global name_convert
	global current_timestamp
	total_test_cases =0
	total_failures=0

	for i in range(len(jdata[0]["suites"])):
		for j in range(len(jdata[0]["suites"][i]["cases"])):
			total_test_cases+=1
			full_name= jdata[0]["suites"][i]["cases"][j]["className"] + jdata[0]["suites"][i]["cases"][j]["name"]
			primary_id=name_convert[full_name]
			if(jdata[0]["suites"][i]["cases"][j]["status"]=="FAILED"):
				x=1
				total_failures+=1
			else:
				x=0
			final_file.write(str(serial_no) +';'+ str(primary_id)+';'+ str(jdata[0]["suites"][i]["cases"][j]["duration"]) +';'+ '0' + ';'+ str(lastrun[primary_id]) + ';'+str(last_results[primary_id]) +';' +str(x)+';'+ str(cycle))
			## The Last Run time of the test case for the further builds is stored.
			lastrun[primary_id]=current_timestamp

			## Formulating the last results history
			last_results[primary_id].append(x)
			final_file.write('\n')
			serial_no+=1

	cycle+=1

	## Printing Total test cases and total no of failures in a build
	print("Total Failures")
	print(total_failures)
	print(" Total Test Cases")
	print(total_test_cases)



##The CSV file where the data has to be written after formatting and hashing
final_file = open('hadoop_testing.csv','w')
final_file.write('Id;Name;Duration;CalcPrio;LastRun;LastResults;Verdict;Cycle;Priority\n')

## Anonymizing the test case names and giving ids to them. idd can start from any number
idd=10000
serial_no=1

## We need only certain fields from the Dump files and that too in a certain format. 
## Dictionaries are created so that the test history and the last run and the idd can be mapped to the test name.
priority=dict()
name_convert =dict()
last_results=dict()
lastrun = dict()

## It keeps track of the CI cycle
cycle = 1
initial_timestamp= "2017-07-17 00:00:01"
current_timestamp = 0

## The Start build and the end build store the build no from which we want to extract the data and the end build no stores the build upto which we want to extract the data
start_build_no = 149
end_build_no = 516


build_no = start_build_no


## The project name and the build_no is specified here
project_name = "hadoop-trunk-win"

while(build_no < end_build_no):	
	url = "https://builds.apache.org/view/All/job/"+str(project_name)+"/"+str(build_no)+"/testReport/api/json?pretty=true/"
	url_timestamp = "https://builds.apache.org/view/All/job/"+str(project_name)+"/"+str(build_no)+"/api/json?pretty=true/"
	
	try:
		## myResponse has the test case build data in the JSON format
		myResponse = requests.get(url)
		response_for_timestamp =requests.get(url_timestamp)
		flag = 1
		
	except:
		## If the URL has failed to connect
		flag = 0

	if(flag):
		jData=[]
		if(myResponse.ok):
			## Getting the timestamp of the build
			jtimestamp=json.loads(response_for_timestamp.content,strict=False)
			timestamp = jtimestamp["timestamp"]
			## Converting the timestamp into a date format
			current_timestamp = datetime.datetime.fromtimestamp(timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')
			# Loads (Load String) takes a Json file and converts into python data structure (dict or list, depending on JSON)
			jData = json.loads(myResponse.content,strict=False)
			jdata=[]
			jdata.append(jData)

			# convert_name_to_idd(jdata)
			# write_to_csv(jdata)

	build_no+=1
	print(build_no)
	print(cycle)
