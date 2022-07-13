def insert_secrets():
   with open("Projeto-01/secrets_reddit.txt", "r") as text:
      lines = text.read(),

   client_id = lines[1]
   client_secret = lines[3]
   password = lines[5]
   user_agent = lines[7]
   username = lines[9]
   
   return client_id, client_secret, password, user_agent, username


with open("Projeto-01/secrets_reddit.txt", "r") as text:
   lines = text.read()

client_id = lines[1]
client_secret = lines[3]
password = lines[5]
user_agent = lines[7]
username = lines[9]

print(client_id)