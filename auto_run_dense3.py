import os
from Dense3_nn import NNmodel

node_list = []
moment = 0.0
for m in range (0,1,1):
    moment+= 0.1
    print ("use momentum =" + str(moment))
    for l in range (0,9,1):
        accuracy_list = [None] * 40
        for i in range(3, 40, 1):
            nodelist = node_list[:]
            nodelist.append(i)
            accuracy = NNmodel(moment,nodelist)
            accuracy_list[i] = accuracy
        m = max(accuracy_list)

        max_accuracy_position = accuracy_list.index(m)
        temp_list = node_list[:]
    #find the max # of node in layers
        if l>0 :
            print (i)
            max_node = max(node_list)
            print("node_list")
            print ( node_list)
            print("max_node")
            print (max_node)
            max_node_position = node_list.index(max_node)
            print("max_node_position")
            print (max_node_position)
            print ("max accuracy")
            print (m)
            while 1:
                temp_list[max_node_position] -= 1
                input_node_list = temp_list[:]
                input_node_list.append(max_accuracy_position)
                print("input_node_list")
                print(input_node_list)
                temp_accuracy = NNmodel(moment,input_node_list)
                if temp_accuracy >= m:
                    print ("temp_accuracy is larger")
                    print (temp_accuracy)
                    m = temp_accuracy
                else:
                    temp_list[max_node_position] +=1
                    break
        else:
            max_node = 0
            max_node_position = 0

        print("temp_list")
        print(temp_list)

        if l>0:
            node_list[max_node_position] = temp_list[max_node_position]

        node_list.append(max_accuracy_position)
        print (max_accuracy_position)

    result = open("final_node.txt", 'a')
    for i in range(0,len(node_list),1):
        result.write(str(i+1)+" DNSE "+str(nodelist[i])+" node\n")
    result.write("\n")


