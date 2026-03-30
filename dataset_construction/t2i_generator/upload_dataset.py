from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your dataset files
upload_folder(folder_path="/fs/projects/SGH_CR_RAI-AP_szh-hpc_users/workplace/mot2sgh/ImageAttributionBench", repo_id="multiitsuki/ImageAttributionBench", repo_type="dataset")
