#it returns array of multiclass prediction results of user's images

# Get the list of all files and directories
# path = "facer_dir/barcelona/"
# dir_list = os.listdir(path)
# results = []
# predict.load_model_func()
# count = 0
# for image in dir_list:
#     output = predict.predict(path+image)
#
#     results.append(output)
#     count+=1
#     print(count)
#
#
# with open('world_results/multi/Barcelona.txt', 'w') as filehandle:
#     json.dump(results, filehandle)



with open("world_results/binary/Barcelona.txt") as f:
    lines = f.readlines()
    str = lines[0]
    arr = (str[1:len(str)-1]).split(", ")
    print(arr)

#emotion_prediction()