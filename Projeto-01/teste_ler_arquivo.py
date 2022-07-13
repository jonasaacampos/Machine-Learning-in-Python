def insert_secret():
    
    list_info = []

    f = open("C:\github\Machine-Learning-in-Python-C\Projeto-01\secrets_reddit.txt", "r")
    for x in f:
      list_info.append(x)
    f.close()

    client_id = list_info[1]
    client_secret = list_info[3]
    password = list_info[5]
    user_agent = list_info[7]
    username = list_info[9]
     
    return client_id, client_secret, password, user_agent, username
    



print(insert_secret())
