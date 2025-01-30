/*
 * Anserini: A Lucene toolkit for reproducible information retrieval research
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package io.anserini.encoder.dense;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class ArcticEmbedLEncoder extends DenseEncoder {
  static private final String MODEL_URL = "file:///home/v4zhong/.cache/pyserini/encoders/arctic-embed-l.onnx";
  static private final String VOCAB_URL = "file:///home/v4zhong/.cache/pyserini/encoders/arctic-embed-l-vocab.txt";

  static private final String MODEL_NAME = "arctic-embed-l.onnx";
  static private final String VOCAB_NAME = "arctic-embed-l-vocab.txt";
  static private final int MAX_SEQ_LEN = 512;

  public ArcticEmbedLEncoder() throws IOException, OrtException, URISyntaxException {
    super(MODEL_NAME, MODEL_URL, VOCAB_NAME, VOCAB_URL);
  }

  @Override
  public float[] encode(String query) throws OrtException {
    if (query == null) {
      throw new IllegalArgumentException("Query cannot be null");
    }

    List<String> queryTokens = new ArrayList<>();
    queryTokens.add("[CLS]");
    queryTokens.addAll(this.tokenizer.tokenize(query));
    queryTokens.add("[SEP]");

    // Truncate if needed
    if (queryTokens.size() > MAX_SEQ_LEN) {
      queryTokens = queryTokens.subList(0, MAX_SEQ_LEN);
    }

    Map<String, OnnxTensor> inputs = new HashMap<>();
    long[] queryTokenIds = convertTokensToIds(this.tokenizer, queryTokens, this.vocab);
    long[][] inputTokenIds = new long[1][queryTokenIds.length];
    inputTokenIds[0] = queryTokenIds;

    // Add attention mask and token type IDs
    long[][] attentionMask = new long[1][queryTokenIds.length];
    long[][] tokenTypeIds = new long[1][queryTokenIds.length];
    Arrays.fill(attentionMask[0], 1);

    inputs.put("input_ids", OnnxTensor.createTensor(environment, inputTokenIds));
    inputs.put("token_type_ids", OnnxTensor.createTensor(environment, tokenTypeIds));
    inputs.put("attention_mask", OnnxTensor.createTensor(environment, attentionMask));

    float[] weights = null;
    try (OrtSession.Result results = this.session.run(inputs)) {
      weights = ((float[][]) results.get("pooler_output").get().getValue())[0];
      weights = normalize(weights);
    } catch (OrtException e) {
      e.printStackTrace();
      throw e;
    }
    return weights;
  }
}
