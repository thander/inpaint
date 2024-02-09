POD_ID=`
	runpodctl \
	create pod \
	--templateId '1iohg19etg' \
	--imageName runpod/stable-diffusion:web-ui-10.2.1 \
	--secureCloud \
	--volumeSize 75 \
	--containerDiskSize 30 \
	--name ease \
	--gpuType 'NVIDIA RTX A4000' \
	--ports '8888/http,3001/http,22/tcp' \
`;

if [[ $POD_ID ]]; then
POD_ID=`echo "$POD_ID" | cut -d'"' -f 2`;
/usr/bin/open -a "/Applications/Google Chrome.app" "https://$POD_ID-3001.proxy.runpod.net/"
read -s -n 1 -p "Press any key to remove pod . . .";

runpodctl remove pod $POD_ID;
fi

