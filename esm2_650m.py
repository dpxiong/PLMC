import os, sys, torch, esm, pickle
import numpy as np

dataset = sys.argv[1]

for d in ['TR', 'TE']:
    data = []
    with open('./data/'+dataset+'/'+d+'_Sequence.fasta', 'r') as infile:
        lines = infile.readlines()
        for i in range(0, len(lines), 2):
            data.append((lines[i].strip()[1:], lines[i+1].strip()))

    result_dir = './data/'+dataset+'/'+d+'_esm2_650m'
    if os.path.exists(result_dir):
        os.system('rm -rf '+result_dir)
    os.system('mkdir '+result_dir)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = model.to(device)
    model.eval()

    for i in data:
        if not os.path.exists(result_dir+'/'+i[0]+'.pkl'):
            print(i[0])
            batch_labels, batch_strs, batch_tokens = batch_converter([i])
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

            # Extract per-residue representations
            with torch.no_grad():
        #         results = model(batch_tokens.to(device), repr_layers=[33], return_contacts=True)
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            token_representations = results['representations'][33]

            with open(result_dir+'/'+i[0]+'.pkl', 'wb') as outfile:
                pickle.dump(np.squeeze(token_representations.detach().cpu().numpy(), axis=0), outfile)
        
print('done!')
