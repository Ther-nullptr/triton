nvcc --resource-usage -Xptxas -dlcm=cv -Xptxas -dscm=wt -std=c++17 \
     -gencode=arch=compute_80,code=\"sm_80,compute_80\" \
      template_shared_lat.cu -o template_shared_lat -I./ -L -lcudart