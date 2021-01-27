import moses

data = open("valid-molecules/Jan-08-2021_2140.txt", "r").read()
data = data.split("\n")


metrics = moses.get_all_metrics(data)
