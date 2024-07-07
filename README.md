## End to End Machine Learning Project - Regression

Description: An end-to-end regression solution implemented using a modular programming approach. The model (with an accuracy of 84%) predicts students' exam performance based on several highly correlated features like parental background, course completion, scores in other sections and more. Deployed on AWS Elastic Beanstalk using CI code pipeline

![image](https://github.com/ferozk0333/MLProject/assets/48884151/d1987c13-bbde-46e8-b54e-90672b7464a3)
![image](https://github.com/ferozk0333/MLProject/assets/48884151/9a279039-6a3c-4bd7-a3a9-c46485ad3839)
![image](https://github.com/ferozk0333/MLProject/assets/48884151/7152dc60-b789-4ed6-8fd5-726f8be5d801)
![image](https://github.com/ferozk0333/MLProject/assets/48884151/bc29d5ea-9ca8-488b-afa9-aad788a5fb34)


https://github.com/ferozk0333/MathMentor-Machine_Learning_based_Test_Score_Predictor/assets/48884151/d5b1190d-caf7-403a-8dfc-97c715407505


## Docker Setup In EC2 commands to be Executed
#optinal

sudo apt-get update -y

sudo apt-get upgrade

#required

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

## Configure EC2 as self-hosted runner:
## Setup github secrets:
AWS_ACCESS_KEY_ID=

AWS_SECRET_ACCESS_KEY=

AWS_REGION = us-east-1

AWS_ECR_LOGIN_URI = demo>> 566373416292.dkr.ecr.ap-south-1.amazonaws.com

ECR_REPOSITORY_NAME = <repo name>








