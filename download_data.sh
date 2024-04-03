if test -f ".zenodo" ; then
  printf "One time data download has already been downloaded, skipping.\n"
  printf "Previously downloaded data can be found in ./Interfuser_data, ./TCP_data, ./LAV_data\n"
else
#    printf "Downloading Interfuser Data (3.5 GB)\n" && wget https://zenodo.org/records/10627823/files/Interfuser_data.7z && 7z x Interfuser_data.7z -o./Interfuser_data && rm ./Interfuser_data.7z && \
    printf "Downloading TCP Data (2.5 GB)\n" && wget https://zenodo.org/records/10627823/files/TCP_data.7z && 7z x TCP_data.7z -o./TCP_data && rm ./TCP_data.7z && \
    printf "Downloading LAV Data (2.6 GB)\n" && wget https://zenodo.org/records/10627823/files/LAV_data.7z && 7z x LAV_data.7z -o./LAV_data && rm ./LAV_data.7z && \
    touch .zenodo && printf "Finished at $(date)\n"
fi