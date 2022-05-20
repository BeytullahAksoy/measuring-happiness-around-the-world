
import os
import pandas as pd
import numpy as np


def multi_visualization(file_path):


    # with open(file_path) as f:
    #     lines = f.readlines()
    #     str = lines[0]
    #     arr = (str[1:len(str)-1]).split(", ")
    #     return arr

    files = os.listdir(file_path)
    cities_name = []
    cities_count = []
    emotions = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    result_count = []
    for file in files:
        with open(file_path+"/"+file) as f:
            lines = f.readlines()
            str = lines[0]
            result = (str[1:len(str)-1]).split(", ")
            result = [int(i) for i in result]
        for i in range(7):
            c = result.count(i)
            result_count.append(c)
        cities_count.append(result_count)
        cities_name.append(file)
        result_count = []


    print(cities_name)
    print(cities_count)
    i = 0
    total_dataframes = []
    for i  in range(len(cities_name)):
        results_df = pd.DataFrame(
        {'City': cities_name[i][:len(cities_name[i])-4],
         'Emotion': emotions,
         'Count': cities_count[i]

         })

        total_dataframes.append(results_df)
    print()
    result_df = total_dataframes[0]
    for i  in range(1,len(total_dataframes)):
        result_df = pd.concat([result_df,total_dataframes[i]])
    result_df = result_df.reset_index()
    result_df = result_df.iloc[:, 1:]
    print(result_df)
    # plot = result_df.plot.bar(x='City', y='Count', rot=0)
    return result_df

def binary_visualization(file_path):


    # with open(file_path) as f:
    #     lines = f.readlines()
    #     str = lines[0]
    #     arr = (str[1:len(str)-1]).split(", ")
    #     return arr

    files = os.listdir(file_path)
    cities_name = []
    cities_count = []
    emotions = ["not happy","happy"]
    result_count = []
    for file in files:
        with open(file_path+"/"+file) as f:
            lines = f.readlines()
            str = lines[0]
            result = (str[1:len(str)-1]).split(", ")
            result = [int(i) for i in result]
        for i in range(2):
            c = result.count(i)
            result_count.append(c)
        cities_count.append(result_count)
        cities_name.append(file)
        result_count = []


    #print(cities_name)
    #print(cities_count)
    i = 0
    total_dataframes = []
    for i  in range(len(cities_name)):
        results_df = pd.DataFrame(
        {'City': cities_name[i][:len(cities_name[i])-4],
         'Emotion': emotions,
         'Count': cities_count[i]

         })

        total_dataframes.append(results_df)
    #print()
    result_df = total_dataframes[0]
    for i  in range(1,len(total_dataframes)):
        result_df = pd.concat([result_df,total_dataframes[i]])
    result_df = result_df.reset_index()
    result_df = result_df.iloc[:, 1:]
    print(result_df.iloc[0].values)


    # plot = result_df.plot.bar(x='City', y='Count', rot=0)
    return result_df



binary_visualization("world_results/binary")