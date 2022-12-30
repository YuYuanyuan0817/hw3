import argparse
parser=argparse.ArgumentParser()
parser.add_argument('-n',dest="name")
parser.add_argument('-p',dest="params")
args= parser.parse_args()
print (f"name:{args.name},params:{args.params}")





    