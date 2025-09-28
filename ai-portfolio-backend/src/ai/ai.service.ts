import { Injectable } from '@nestjs/common';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { ChatGroq } from '@langchain/groq';
import { pipeline } from '@xenova/transformers';
import { Document } from 'langchain/document';

let vectorStore: MemoryVectorStore | null = null;

@Injectable()
export class AiService {
  private embeddings: any;

  /** Initialize RAG system */
  async initRAG() {
    const docs = [
      { text: 'I am Jahanzaib, a developer skilled in Frontend, Backend, RAG, and CI/CD.' },
      { text: 'I build AI apps with LangChain, Vector Stores, and deploy on VPS with LXC.' },
      { text: 'I have 6 years of experience in development.' },
    ];

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 200,
      chunkOverlap: 50,
    });

    const splitDocs = await splitter.splitDocuments(
      docs.map((d) => new Document({ pageContent: d.text, metadata: {} })),
    );

    // ✅ Load Xenova embeddings (runs locally, no API key required)
    if (!this.embeddings) {
      this.embeddings = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
    }

    // Wrap Xenova pipeline in LangChain-compatible Embeddings interface
    const embedder = {
      embedQuery: async (text: string): Promise<number[]> => {
        const output = await this.embeddings(text, { pooling: 'mean', normalize: true });
        return Array.from(output.data);
      },
      embedDocuments: async (texts: string[]): Promise<number[][]> => {
        const results: number[][] = [];
        for (const t of texts) {
          const output = await this.embeddings(t, { pooling: 'mean', normalize: true });
          results.push(Array.from(output.data));
        }
        return results;
      },
    };

    vectorStore = await MemoryVectorStore.fromDocuments(splitDocs, embedder);
  }

  /** Ask a question using RAG + Groq LLM */
  async askAI(question: string) {
    if (!vectorStore) {
      await this.initRAG();
    }
    if (!vectorStore) throw new Error('VectorStore initialization failed');

    const relevantDocs = await vectorStore.similaritySearch(question, 2);
    const context = relevantDocs.map((d) => d.pageContent).join('\n');

    const model = new ChatGroq({
      model: 'llama-3.3-70b-versatile', // or 'llama3-70b-8192'
      apiKey: process.env.GROQ_API_KEY,
    });

    const response = await model.invoke([
      { role: 'system', content: 'You are a portfolio assistant for Jahanzaib.' },
      { role: 'user', content: `Context:\n${context}\n\nQuestion: ${question}` },
    ]);

    return { answer: response.content };
  }
}
