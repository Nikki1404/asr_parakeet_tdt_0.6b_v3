C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\pydub\utils.py:170: RuntimeWarning: Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work
  warn("Couldn't find ffmpeg or avconv - defaulting to ffmpeg, but may not work", RuntimeWarning)
================================================================================
TOTAL FILES = 56
WORKERS = 2
SPEED = 4.0x
================================================================================

[1/56] PROCESSING herring1.mp3

STARTING -> herring1.mp3
C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\pydub\utils.py:198: RuntimeWarning: Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work
  warn("Couldn't find ffprobe or avprobe - defaulting to ffprobe, but may not work", RuntimeWarning)

[2/56] PROCESSING herring10.mp3

STARTING -> herring10.mp3

[3/56] PROCESSING herring11.mp3

STARTING -> herring11.mp3

[4/56] PROCESSING herring12.mp3

STARTING -> herring12.mp3

[5/56] PROCESSING herring13.mp3

STARTING -> herring13.mp3

[6/56] PROCESSING herring14.mp3

STARTING -> herring14.mp3

[7/56] PROCESSING herring15.mp3

STARTING -> herring15.mp3

[8/56] PROCESSING herring16.mp3

STARTING -> herring16.mp3

[9/56] PROCESSING herring17.mp3

STARTING -> herring17.mp3

[10/56] PROCESSING herring2.mp3

STARTING -> herring2.mp3

[11/56] PROCESSING herring3.mp3

STARTING -> herring3.mp3

[12/56] PROCESSING herring5.mp3

STARTING -> herring5.mp3

[13/56] PROCESSING herring6.mp3

STARTING -> herring6.mp3

[14/56] PROCESSING herring7.mp3

STARTING -> herring7.mp3

[15/56] PROCESSING herring8.mp3

STARTING -> herring8.mp3

[16/56] PROCESSING herring9.mp3

STARTING -> herring9.mp3

[17/56] PROCESSING maria1.mp3

STARTING -> maria1.mp3

[18/56] PROCESSING maria10.mp3

STARTING -> maria10.mp3

[19/56] PROCESSING maria16.mp3

STARTING -> maria16.mp3

[20/56] PROCESSING maria18.mp3

STARTING -> maria18.mp3

[21/56] PROCESSING maria19.mp3

STARTING -> maria19.mp3

[22/56] PROCESSING maria2.mp3

STARTING -> maria2.mp3

[23/56] PROCESSING maria20.mp3

STARTING -> maria20.mp3

[24/56] PROCESSING maria21.mp3

STARTING -> maria21.mp3

[25/56] PROCESSING maria24.mp3

STARTING -> maria24.mp3

[26/56] PROCESSING maria27.mp3

STARTING -> maria27.mp3

[27/56] PROCESSING maria30.mp3

STARTING -> maria30.mp3

[28/56] PROCESSING maria31.mp3

STARTING -> maria31.mp3

[29/56] PROCESSING maria4.mp3

STARTING -> maria4.mp3

[30/56] PROCESSING maria40.mp3

STARTING -> maria40.mp3

[31/56] PROCESSING maria7.mp3

STARTING -> maria7.mp3

[32/56] PROCESSING sastre1.mp3

STARTING -> sastre1.mp3

[33/56] PROCESSING sastre10.mp3

STARTING -> sastre10.mp3

[34/56] PROCESSING sastre11.mp3

STARTING -> sastre11.mp3

[35/56] PROCESSING sastre12.mp3

STARTING -> sastre12.mp3

[36/56] PROCESSING sastre13.mp3

STARTING -> sastre13.mp3

[37/56] PROCESSING sastre2.mp3

STARTING -> sastre2.mp3

[38/56] PROCESSING sastre3.mp3

STARTING -> sastre3.mp3

[39/56] PROCESSING sastre4.mp3

STARTING -> sastre4.mp3

[40/56] PROCESSING sastre5.mp3

STARTING -> sastre5.mp3

[41/56] PROCESSING sastre6.mp3

STARTING -> sastre6.mp3

[42/56] PROCESSING sastre7.mp3

STARTING -> sastre7.mp3

[43/56] PROCESSING sastre8.mp3

STARTING -> sastre8.mp3

[44/56] PROCESSING sastre9.mp3

STARTING -> sastre9.mp3

[45/56] PROCESSING zeledon1.mp3

STARTING -> zeledon1.mp3

[46/56] PROCESSING zeledon11.mp3

STARTING -> zeledon11.mp3

[47/56] PROCESSING zeledon13.mp3

STARTING -> zeledon13.mp3

[48/56] PROCESSING zeledon14.mp3

STARTING -> zeledon14.mp3

[49/56] PROCESSING zeledon2.mp3

STARTING -> zeledon2.mp3

[50/56] PROCESSING zeledon3.mp3

STARTING -> zeledon3.mp3

[51/56] PROCESSING zeledon4.mp3

STARTING -> zeledon4.mp3

[52/56] PROCESSING zeledon5.mp3

STARTING -> zeledon5.mp3

[53/56] PROCESSING zeledon6.mp3

STARTING -> zeledon6.mp3

[54/56] PROCESSING zeledon7.mp3

STARTING -> zeledon7.mp3

[55/56] PROCESSING zeledon8.mp3

STARTING -> zeledon8.mp3

[56/56] PROCESSING zeledon9.mp3

STARTING -> zeledon9.mp3
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 276, in <module>
    asyncio.run(
    ~~~~~~~~~~~^
        run_batch(
        ^^^^^^^^^^
    ...<4 lines>...
        )
        ^
    )
    ^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 195, in run
    return runner.run(main)
           ~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^
  File "C:\Program Files\Python313\Lib\asyncio\base_events.py", line 725, in run_until_complete
    return future.result()
           ~~~~~~~~~~~~~^^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 239, in run_batch
    await asyncio.gather(*tasks)
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 205, in worker
    await transcribe_file(filepath, host, port, speed)
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 67, in transcribe_file
    pcm, duration_sec = load_mp3_as_pcm16(filepath)
                        ~~~~~~~~~~~~~~~~~^^^^^^^^^^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\benchmarking_client.py", line 31, in load_mp3_as_pcm16
    audio = AudioSegment.from_mp3(filepath)
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\pydub\audio_segment.py", line 796, in from_mp3
    return cls.from_file(file, 'mp3', parameters=parameters)
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\pydub\audio_segment.py", line 728, in from_file
    info = mediainfo_json(orig_file, read_ahead_limit=read_ahead_limit)
  File "C:\Users\re_nikitav\Documents\parakeet-asr-multilingual\venv\Lib\site-packages\pydub\utils.py", line 274, in mediainfo_json
    res = Popen(command, stdin=stdin_parameter, stdout=PIPE, stderr=PIPE)
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1038, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
    ~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                        pass_fds, cwd, env,
                        ^^^^^^^^^^^^^^^^^^^
    ...<5 lines>...
                        gid, gids, uid, umask,
                        ^^^^^^^^^^^^^^^^^^^^^^
                        start_new_session, process_group)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Program Files\Python313\Lib\subprocess.py", line 1550, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
                             # no special security
                             ^^^^^^^^^^^^^^^^^^^^^
    ...<4 lines>...
                             cwd,
                             ^^^^
                             startupinfo)
                             ^^^^^^^^^^^^
FileNotFoundError: [WinError 2] The system cannot find the file specified
