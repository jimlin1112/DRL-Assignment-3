file = open("output.txt", "w")
save_path = "model.pth"

fin = open("record.txt", "r")
for line in fin.readlines():
    file.write(line)
fin.close()