import { Embeddings, EmbeddingsParams } from "@langchain/core/embeddings";
import type { AsyncCallerParams } from "@langchain/core/utils/async_caller";
import { pipeline } from "@xenova/transformers";

export interface XenovaEmbeddingsParams extends EmbeddingsParams, AsyncCallerParams {
  modelName?: string;
}

export class XenovaEmbeddings extends Embeddings implements XenovaEmbeddingsParams {
  modelName: string;
  private model: any;

  constructor(fields?: XenovaEmbeddingsParams) {
    super(fields ?? {}); // ✅ Pass fields to parent (Embeddings expects caller config)
    this.modelName = fields?.modelName ?? "Xenova/all-MiniLM-L6-v2";
  }

  private async loadModel() {
    if (!this.model) {
      this.model = await pipeline("feature-extraction", this.modelName);
    }
  }

  async embedDocuments(texts: string[]): Promise<number[][]> {
    await this.loadModel();
    const results: number[][] = [];

    for (const text of texts) {
      const output = await this.model(text, { pooling: "mean", normalize: true });
      results.push(Array.from(output.data));
    }

    return results;
  }

  async embedQuery(text: string): Promise<number[]> {
    await this.loadModel();
    const output = await this.model(text, { pooling: "mean", normalize: true });
    return Array.from(output.data);
  }
}
