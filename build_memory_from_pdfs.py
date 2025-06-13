#!/usr/bin/env python3
"""
Extrai texto de todos os PDFs em uma pasta e cria a memória memvid completa (v2.1).
"""
import sys
import os
import time
from pypdf import PdfReader

# Adiciona o diretório raiz ao path para encontrar a biblioteca memvid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memvid import MemvidEncoder
# Importamos a configuração para saber a extensão de vídeo padrão
from memvid.config import get_codec_parameters, VIDEO_CODEC

def extract_text_from_pdfs(pdf_folder: str) -> str:
    """Lê todos os arquivos PDF em uma pasta e retorna um único string de texto."""
    all_text = ""
    print(f"Buscando PDFs na pasta: {pdf_folder}")
    
    # Verifica se a pasta existe e tem arquivos
    if not os.path.isdir(pdf_folder) or not os.listdir(pdf_folder):
        return ""
        
    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_folder, filename)
            print(f"  -> Extraindo texto de: {filename}")
            try:
                reader = PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n"
            except Exception as e:
                print(f"    [AVISO] Não foi possível ler o arquivo {filename}: {e}")
    return all_text

def main():
    PDF_SOURCE_FOLDER = "papers"
    OUTPUT_DIR = "output"
    
    if not os.path.isdir(PDF_SOURCE_FOLDER):
        os.makedirs(PDF_SOURCE_FOLDER)
        print(f"Pasta '{PDF_SOURCE_FOLDER}' criada.")
        print("Por favor, coloque seus arquivos PDF dentro dela e rode o script novamente.")
        return

    print("Iniciando extração de texto dos PDFs...")
    full_text = extract_text_from_pdfs(PDF_SOURCE_FOLDER)
    
    if not full_text:
        print(f"Nenhum texto foi extraído. Verifique se há arquivos .pdf na pasta '{PDF_SOURCE_FOLDER}'.")
        return
        
    print(f"\nExtração concluída. Total de {len(full_text)} caracteres lidos.")
    print("=" * 50)
    
    print("Memvid: Construindo a Memória em Vídeo")
    print("=" * 50)
    
    # Para o Raspberry Pi, podemos usar um codec mais leve como o 'mp4v'
    # que usa o OpenCV nativo e não precisa de FFmpeg ou Docker.
    # Se quiser máxima compressão, pode usar 'h265' (pode ser mais lento).
    # Vamos usar 'mp4v' para garantir compatibilidade e velocidade no RPi.
    codec_selecionado = 'mp4v' 
    
    encoder = MemvidEncoder()
    
    print("\nAdicionando texto com chunking automático...")
    # Ajustamos o chunk_size para um valor menor, ideal para RAG
    encoder.add_text(full_text, chunk_size=512, overlap=50)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- CORREÇÃO PRINCIPAL AQUI ---
    # Define o nome completo dos arquivos, incluindo a extensão do vídeo.
    # Usamos a configuração do memvid para obter a extensão correta para o codec.
    video_extension = get_codec_parameters(codec_selecionado)["video_file_type"]
    video_file = os.path.join(OUTPUT_DIR, f"memory.{video_extension}")
    index_file = os.path.join(OUTPUT_DIR, "memory_index.json")
    # -----------------------------
    
    print(f"\nConstruindo vídeo: {video_file} (Codec: {codec_selecionado})")
    print(f"Construindo índice: {index_file}")
    
    start_time = time.time()
    # Passamos o codec escolhido para a função build_video
    build_stats = encoder.build_video(video_file, index_file, codec=codec_selecionado, show_progress=True)
    elapsed = time.time() - start_time
    
    print(f"\nConstrução concluída em {elapsed:.2f} segundos")
    print(f"Total de chunks processados: {build_stats['index_stats']['total_chunks']}")
    print(f"Tamanho do vídeo: {build_stats['video_size_mb']:.2f} MB")
    print("\nSucesso! Memória criada com os seguintes arquivos:")
    print(f"  - {video_file}")
    print(f"  - {index_file}")
    print(f"  - {index_file.replace('.json', '.faiss')}")
    print("\nAgora você pode rodar a API FastAPI com 'python main.py'")

if __name__ == "__main__":
    main()