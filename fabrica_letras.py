"""Fábrica de letras do Sitonize.

Fica rodando no PC. Quando alguém toca uma música sem letra, o app grava um pedido em `pedidos_letra`.
Aqui: acha a gravação original no YouTube, a IA (Whisper) escuta e escreve a letra com o tempo de cada frase,
e a letra vai para `letras_cache`, de onde todos os celulares leem.

Uso:  python fabrica_letras.py          (fica em loop)
      python fabrica_letras.py --uma    (processa o que houver e sai)
"""
import base64, json, os, re, subprocess, sys, tempfile, time, unicodedata
import psycopg2
from faster_whisper import WhisperModel

SENHA = os.environ.get('DBPASS') or open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.dbpass')).read().strip()
SO_COM_AUDIO = bool(os.environ.get('SO_COM_AUDIO'))  # na nuvem: só pedidos cujo celular já deixou o áudio
MODELO = os.environ.get('MODELO', 'large-v3')
MARCA_KARAOKE = re.compile(r'karaok|instrumental|playback|play back|sem voz|backing track|minus one|sing along|\bletra\b|lyrics|oficial|official|audio|áudio|ao vivo|clipe|video|vídeo|\([^)]*\)|\[[^\]]*\]', re.I)


def conecta():
    return psycopg2.connect(host='aws-1-sa-east-1.pooler.supabase.com', port=5432, user='postgres.soyxccghiqpgrjdogqmw',
                            password=SENHA, dbname='postgres', sslmode='require')


def limpa(t):
    t = MARCA_KARAOKE.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip(' -–|')


def baixa_original(titulo, artista, dur_ms, pasta):
    """Procura a gravação original (não karaokê) com duração parecida e baixa o áudio em wav 16 kHz mono."""
    consulta = limpa(titulo) + ' ' + artista
    r = subprocess.run(['yt-dlp', '--dump-json', '--flat-playlist', f'ytsearch6:{consulta} audio oficial'],
                       capture_output=True, text=True, encoding='utf-8')
    print('  busca:', r.returncode, len(r.stdout.splitlines()), (r.stderr or '')[-400:])
    cand = []
    for l in r.stdout.splitlines():
        try:
            j = json.loads(l)
        except Exception:
            continue
        print('   cand:', j.get('duration'), j.get('title'))
        if re.search(r'karaok|instrumental|playback|cover|sem voz', (j.get('title') or ''), re.I):
            continue
        d = (j.get('duration') or 0) * 1000
        if dur_ms and d and abs(d - dur_ms) > 25000 and abs(d - dur_ms) > 0.15 * dur_ms:
            continue
        cand.append((abs(d - dur_ms) if dur_ms and d else 0, j))
    if not cand:
        print('  sem candidato (dur_ms=%s)' % dur_ms)
        return None, 0
    cand.sort(key=lambda x: x[0])
    j = cand[0][1]
    saida = os.path.join(pasta, 'a.%(ext)s')
    d = None
    for cli in ('android_vr', 'tv', 'mweb', 'ios', 'web_safari', 'tv_embedded'):
        d = subprocess.run(['yt-dlp', '--extractor-args', 'youtube:player_client=' + cli, '-f', 'bestaudio', '-o', saida,
                            'https://www.youtube.com/watch?v=' + j['id']], capture_output=True, text=True)
        print('  cliente', cli, '->', d.returncode, (d.stderr or '').strip().splitlines()[-1][:160] if d.returncode and d.stderr else '')
        if d.returncode == 0:
            break
    if d.returncode:
        return None, 0
    bruto = [f for f in os.listdir(pasta) if f.startswith('a.')][0]
    wav = os.path.join(pasta, 'a16.wav')
    subprocess.run(['ffmpeg', '-y', '-i', os.path.join(pasta, bruto), '-ac', '1', '-ar', '16000', wav], check=True, capture_output=True)
    return wav, int((j.get('duration') or 0) * 1000)


# frases que o Whisper inventa em trechos sem voz (créditos de legenda, "obrigado por assistir"...)
INVENTADA = re.compile(r'[Ѐ-ӿ؀-ۿ぀-ヿ一-鿿가-힯]|god bless you|thank you for watching|thanks for watching|^\W*(m[úu]sica|music|aplausos|applause)\W*$|legenda|legendas|obrigad[oa] por assistir|inscreva|amara\.org|tradu[çc][ãa]o|subt[íi]tulos?|www\.|\.com', re.I)


def escreve(modelo, wav, idioma=None):
    import wave
    import numpy as np
    with wave.open(wav, 'rb') as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if not idioma:
        # o Whisper erra o idioma em trechos com pouca voz (hino virou "russo"); sem certeza forte, assume português
        try:
            lg, pr, todas = modelo.detect_language(audio)
            probs = dict(todas)
            idioma = lg if lg in ('pt', 'en', 'es') and pr >= 0.85 and probs.get('pt', 0) < 0.2 else 'pt'
            print('  idioma detectado', lg, round(pr, 2), '-> usando', idioma)
        except Exception as e:
            print('  detecção de idioma falhou:', e)
            idioma = 'pt'
    seg, info = modelo.transcribe(audio, language=idioma, vad_filter=False, condition_on_previous_text=False, beam_size=5,
                                  word_timestamps=True, hallucination_silence_threshold=1.5, temperature=[0.0, 0.2, 0.4],
                                  compression_ratio_threshold=2.0, no_speech_threshold=0.6, chunk_length=15)
    linhas = []
    for s in seg:
        t = s.text.strip()
        if not t or INVENTADA.search(t):
            continue
        linhas.append([int(s.start * 1000), t])
    return linhas, info.language


def audio_do_banco(db, chave, pasta):
    """O celular que pediu a letra deixa o áudio da gravação original no banco (servidor não consegue baixar do YouTube)."""
    with db.cursor() as c:
        c.execute("select b64, container from pedidos_audio where chave=%s", (chave,))
        r = c.fetchone()
    if not r:
        return None
    bruto = os.path.join(pasta, 'a.' + (r[1] or 'bin'))
    with open(bruto, 'wb') as f:
        f.write(base64.b64decode(r[0]))
    wav = os.path.join(pasta, 'a16.wav')
    subprocess.run(['ffmpeg', '-y', '-i', bruto, '-ac', '1', '-ar', '16000', wav], check=True, capture_output=True)
    return wav


def processa(db, modelo, p):
    chave, video, titulo, artista, dur = p
    with tempfile.TemporaryDirectory() as pasta:
        wav = audio_do_banco(db, chave, pasta)
        dur_orig = 0
        if wav:
            print('  áudio veio do celular')
        else:
            wav, dur_orig = baixa_original(titulo, artista, dur, pasta)
        if not wav:
            return False
        linhas, idioma = escreve(modelo, wav)
    # frases repetidas em sequência = Whisper travado; tira
    lim = []
    for l in linhas:
        if lim and lim[-1][1].lower() == l[1].lower() and len(lim) > 2 and lim[-2][1].lower() == l[1].lower():
            continue
        lim.append(l)
    print(f'  {len(lim)} linhas ({idioma})')
    if len(lim) < 8:
        return False
    with db.cursor() as c:
        c.execute("insert into letras_cache (chave, linhas, sincronizada, duracao_ms, fonte) values (%s,%s,true,%s,'whisper') "
                  "on conflict (chave) do nothing", (chave, json.dumps(lim), dur_orig or dur))
    db.commit()
    return True


def main():
    uma = '--uma' in sys.argv
    modelo = WhisperModel(MODELO, device='cpu', compute_type='int8', cpu_threads=int(os.environ.get('THREADS', '10')))
    while True:
        db = conecta()
        with db.cursor() as c:
            c.execute("select chave, video, titulo, artista, dur_ms from pedidos_letra where estado='novo' and tentativas < 3 "
                      "and not exists (select 1 from letras_cache l where l.chave = pedidos_letra.chave) "
                      + ("and exists (select 1 from pedidos_audio a where a.chave = pedidos_letra.chave) " if SO_COM_AUDIO else "")
                      + "order by criado limit 1")
            p = c.fetchone()
        if not p:
            db.close()
            if uma:
                return
            time.sleep(30)
            continue
        print('pedido:', p[0], '|', p[2])
        with db.cursor() as c:
            c.execute("update pedidos_letra set tentativas = tentativas + 1 where chave=%s", (p[0],))
        db.commit()
        try:
            ok = processa(db, modelo, p)
        except Exception as e:
            print('  erro:', e)
            ok = False
        with db.cursor() as c:
            c.execute("update pedidos_letra set estado=%s where chave=%s", ('pronto' if ok else 'novo', p[0]))
            if ok:
                c.execute("delete from pedidos_audio where chave=%s", (p[0],))  # não guarda áudio depois de usar
            c.execute("delete from pedidos_audio where criado < now() - interval '2 days'")
        db.commit()
        db.close()


if __name__ == '__main__':
    main()
