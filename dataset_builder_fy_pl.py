import soundfile as sf
from datasets import load_dataset
import csv
import datasets 
import os
import re


from huggingface_hub import login
login(token="hf_")



#n_max_train = 34605
n_max_train = 3000
n_max_valid = 3000

seed = 34253
#fsicoli/common_voice_17_0

dataset_train_ur = load_dataset('mozilla-foundation/common_voice_17_0','fy-NL', split='train', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_train_ur = dataset_train_ur.shuffle(seed=seed)
dataset_train_ur = dataset_train_ur.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_ru = load_dataset('mozilla-foundation/common_voice_17_0','ru', split='train', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_train_ru = dataset_train_ru.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_fa = load_dataset('mozilla-foundation/common_voice_17_0', 'fa', split='train', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_train_fa=dataset_train_fa.shuffle(seed=seed)
#dataset_train_fa = dataset_train_fa.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_train_hi = load_dataset('mozilla-foundation/common_voice_17_0', 'nl', split='train', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_train_hi = dataset_train_hi.shuffle(seed=seed)
dataset_train_hi = dataset_train_hi.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_train_en = load_dataset('mozilla-foundation/common_voice_17_0', 'en', split='train', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_train_en = dataset_train_en.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))





dataset_test_ur = load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='test', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_test_ur = dataset_test_ur.shuffle(seed=seed)
dataset_test_ur = dataset_test_ur.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_fa = load_dataset('mozilla-foundation/common_voice_17_0', 'fa', split='test', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_test_fa = dataset_test_fa.shuffle(seed=seed)
#dataset_test_fa = dataset_test_fa.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_test_hi = load_dataset('mozilla-foundation/common_voice_17_0', 'nl', split='test', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_test_hi = dataset_test_hi.shuffle(seed=seed)
dataset_test_hi = dataset_test_hi.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_sv = load_dataset('mozilla-foundation/common_voice_17_0', 'sv-SE', split='test', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_test_sv = dataset_test_sv.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_test_fy= load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='test', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_test_fy = dataset_test_fy.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))




dataset_validation_ur = load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='validation', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_validation_ur = dataset_validation_ur.shuffle(seed=seed)
dataset_validation_ur = dataset_validation_ur.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_fa =load_dataset('mozilla-foundation/common_voice_17_0', 'fa', split='validation', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_validation_fa = dataset_validation_fa.shuffle(seed=seed)
#dataset_validation_fa = dataset_validation_fa.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

dataset_validation_hi = load_dataset('mozilla-foundation/common_voice_17_0', 'nl', split='validation', streaming=True, trust_remote_code=True) #, use_auth_token=True)
dataset_validation_hi = dataset_validation_hi.shuffle(seed=seed)
dataset_validation_hi = dataset_validation_hi.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_sv =load_dataset('mozilla-foundation/common_voice_17_0', 'sv-SE', split='validation', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_validation_sv = dataset_validation_sv.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))

#dataset_validation_fy =load_dataset('mozilla-foundation/common_voice_17_0', 'fy-NL', split='validation', streaming=True, trust_remote_code=True) #, use_auth_token=True)
#dataset_validation_fy = dataset_validation_fy.cast_column("audio", datasets.features.Audio(sampling_rate=16_000))


#dataset_invalidated_en = load_dataset('mozilla-foundation/common_voice_16_1', 'en', split='invalidated', streaming=True)
#dataset_invalidated_de = load_dataset('mozilla-foundation/common_voice_16_1', 'de', split='invalidated', streaming=True)
#dataset_invalidated_nl =load_dataset('mozilla-foundation/common_voice_16_1', 'nl', split='invalidated', streaming=True)
#dataset_invalidated_sv_SE =load_dataset('mozilla-foundation/common_voice_16_1', 'sv-SE', split='invalidated', streaming=True)
#dataset_invalidated_fy =load_dataset('mozilla-foundation/common_voice_16_1', 'fy-NL', split='invalidated', streaming=True)



#dataset_other_en = load_dataset('mozilla-foundation/common_voice_16_1', 'en', split='other', streaming=True)
#dataset_other_de = load_dataset('mozilla-foundation/common_voice_16_1', 'de', split='other', streaming=True)
#dataset_other_nl= load_dataset('mozilla-foundation/common_voice_16_1', 'nl', split='other', streaming=True)
#datase_other_sv_SE =load_dataset('mozilla-foundation/common_voice_16_1', 'sv-SE', split='other', streaming=True)
#dataset_other_fy=load_dataset('mozilla-foundation/common_voice_16_1', 'fy-NL', split='other', streaming=True)


# list_of_rows_train_nl = list(dataset_train_nl)    # 34605                 # all: 36.4k
# list_of_rows_test_nl = list(dataset_test_nl)             # 11235          # all:11.2k
# list_of_rows_validation_nl = list(dataset_validation_nl)    # 11223       # all:11.2k
# #list_of_rows_invalidated_nl = list(dataset_invalidated_nl.take(5540))            # all: 5.54k
# #list_of_rows_other_nl=list(dataset_other_nl.take(2290))                          # all: 2.29k
# print("nl", len(list_of_rows_train_nl), len(list_of_rows_test_nl), len(list_of_rows_validation_nl))

# list_of_rows_train_en = list(dataset_train_en.take(len(list_of_rows_train_nl)))   #34605            # all: 114k
# list_of_rows_test_en = list(dataset_test_en)           #11235      # all: 16.4k
# list_of_rows_validation_en = list(dataset_validation_en)  # 11223  # all: 16.4k
# #list_of_rows_invalidated_en = list(dataset_invalidated_en.take(5800))     # all: 111k
# #list_of_rows_other_en = list(dataset_other_en.take(5800))                 # all: 154k
# print("en", len(list_of_rows_train_en), len(list_of_rows_test_en), len(list_of_rows_validation_en))


# list_of_rows_train_de = list(dataset_train_de.take(len(list_of_rows_train_nl)))    #34605            # all : 121k
# list_of_rows_test_de = list(dataset_test_de)           #11235       # all:  16.2k
# list_of_rows_validation_de = list(dataset_validation_de) # 11223    # all: 16.2k
# #list_of_rows_invalidated_de = list(dataset_invalidated_de.take(5800))      # all: 53.8k
# #list_of_rows_other_de=list(dataset_other_de.take(5800))                    # all: 7k
# print("de", len(list_of_rows_train_de), len(list_of_rows_test_de), len(list_of_rows_validation_de))


# list_of_rows_train_sv_SE = list(dataset_train_sv_SE) # 7657                # all: 7.66k
# list_of_rows_test_sv_SE = list(dataset_test_sv_SE)          #5206          # all: 5.21k
# list_of_rows_validation_sv_SE = list(dataset_validation_sv_SE)  # 5222     # all: 5.22k
# #list_of_rows_invalidated_sv_SE = list(dataset_invalidated_sv_SE.take(1420))       # all: 1.42k
# #list_of_rows_other_sv_SE=list(dataset_other_sv_SE.take(6610))                     # all: 6.61k
# print("sv-SE", len(list_of_rows_train_sv_SE), len(list_of_rows_test_sv_SE), len(list_of_rows_validation_sv_SE))


# list_of_rows_train_fy = list(dataset_train_fy)   #3920                 # all: 3.92k
# list_of_rows_test_fy=list(dataset_test_fy)       # 3171                # all: 3.17k
# list_of_rows_validation_fy = list(dataset_validation_fy) # 3171        # all:3.17k
# #list_of_rows_invalidated_fy = list(dataset_invalidated_fy.take(3950))         # all: 3.95k
# #list_of_rows_other_fy=list(dataset_other_fy.take(102000))                     # all: 102k
# print("fy-NL", len(list_of_rows_train_fy), len(list_of_rows_test_fy), len(list_of_rows_validation_fy))




#print(list_of_rows_train_de)
whole_rows_train = []
multilingual_dataset_list_train = []
multilingual_dataset_list_train.append([dataset_train_ur,"fy-NL"])
multilingual_dataset_list_train.append([dataset_train_hi,"nl"])
#multilingual_dataset_list_train.append([dataset_train_fa,"fa"])
#multilingual_dataset_list_train.append([dataset_train_sv,"sv-SE"])
#multilingual_dataset_list_train.append([dataset_train_fy,"fy-NL"])


whole_rows_test = []
multilingual_dataset_list_test = []
multilingual_dataset_list_test.append([dataset_test_ur,"fy-NL"])
#multilingual_dataset_list_test.append([dataset_test_fa,"fa"])
multilingual_dataset_list_test.append([dataset_test_hi,"nl"])
#multilingual_dataset_list_test.append([dataset_test_sv,"sv-SE"])
#multilingual_dataset_list_test.append([dataset_test_fy,"fy-NL"])


whole_rows_validation = []
multilingual_dataset_list_validation = []
multilingual_dataset_list_validation.append([dataset_validation_ur,"fy-NL"])
#multilingual_dataset_list_validation.append([dataset_validation_fa,"fa"])
multilingual_dataset_list_validation.append([dataset_validation_hi,"nl"])
#multilingual_dataset_list_validation.append([dataset_validation_sv,"sv-SE"])
#multilingual_dataset_list_validation.append([dataset_validation_fy,"fy-NL"])


whole_rows_test_ur=[]
#whole_rows_test_fa=[]
whole_rows_test_hi=[]
#whole_rows_test_de=[]
#whole_rows_test_nl=[]


#clips_directory = "clips/"
os.makedirs("train_fy_nl/", exist_ok=True)
os.makedirs("test_fy_nl/", exist_ok=True)
os.makedirs("validation_fy_nl/",exist_ok=True)


print("train")
for pair in multilingual_dataset_list_train:
	LID=pair[1]
	print(LID)
	n = 0
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		client_id=row_dict['client_id']
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		whole_rows_train.append([filename, sentence_without_comma, client_id, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("train_fy_nl/"+filename, audio_content, 16000)
		n += 1
		if n == n_max_train:
			break

print("test")
for pair in multilingual_dataset_list_test:
	LID=pair[1]
	print(LID)
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		client_id=row_dict['client_id']
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		whole_rows_test.append([filename, sentence_without_comma, client_id, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("test_fy_nl/"+filename, audio_content, 16000)
		if LID=="fy-NL":
			whole_rows_test_ur.append([filename, sentence_without_comma, client_id, LID])
		elif LID=="nl":
			whole_rows_test_hi.append([filename, sentence_without_comma, client_id, LID])
		#elif LID=="hi":
		#	whole_rows_test_hi.append([filename, sentence_without_comma,client_id, LID])
		#elif LID=="en":
		#	whole_rows_test_en.append([filename, sentence_without_comma, LID])

print("validation")
for pair in multilingual_dataset_list_validation:
	LID=pair[1]
	print(LID)
	n = 0
	for row_dict in iter(pair[0]):
		audio_content=row_dict['audio']['array']
		client_id=row_dict['client_id']
		sentence_without_comma = re.sub(r',', ' ', row_dict['sentence'])
		filename = row_dict['audio']['path'].split("/")[1].replace(".mp3",".wav")
		whole_rows_validation.append([filename, sentence_without_comma, client_id, LID]) #up_votes, down_votes, age, gender, accent, locale, segment, variant
		sf.write("validation_fy_nl/"+filename, audio_content, 16000)
		n += 1
		if n == n_max_valid:
			break


# export splits
with open("train_fy_nl/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_train:
		writer.writerow(row)

with open("test_fy_nl/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_test:
		writer.writerow(row)

with open("validation_fy_nl/metadata.csv","w",newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_validation:
		writer.writerow(row)
'''
# export test files for each language
with open("test_ur.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_test_ur:
		writer.writerow(row)

with open("test_fa.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_test_fa:
		writer.writerow(row)

with open("test_hi.csv","w", newline='') as tsvfile:
	writer = csv.writer(tsvfile, delimiter=',', lineterminator='\n')
	writer.writerow(['file_name','transcription','client_id','lid'])
	for row in whole_rows_test_hi:
		writer.writerow(row)

'''

