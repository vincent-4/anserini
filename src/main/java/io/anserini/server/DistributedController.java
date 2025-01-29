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

package io.anserini.server;

import io.anserini.index.IndexInfo;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;

@RestController
@RequestMapping(path = "/api/v1.1")
public class DistributedController {
    private final Map<String, DistributedSearch> services = new ConcurrentHashMap<>();

    @ResponseStatus(HttpStatus.BAD_REQUEST)
    @ExceptionHandler(IllegalArgumentException.class)
    public Map<String, String> handleIllegalArgumentException(IllegalArgumentException ex) {
        return Map.of("error", ex.getMessage());
    }

    @RequestMapping(method = RequestMethod.GET, path = "/distributed-indexes/{index}/search")
    public Map<String, Object> searchDistributedIndex(
        @PathVariable(value = "index", required = true) String index,
        @RequestParam("query") String query,
        @RequestParam(value = "hits", defaultValue = "10") int hits,
        @RequestParam(value = "qid", defaultValue = "") String qid,
        @RequestParam(value = "efSearch", required = false) Integer efSearch,
        @RequestParam(value = "encoder", required = false) String encoder,
        @RequestParam(value = "queryGenerator", required = false) String queryGenerator,
        @RequestParam(value = "threadsPerShard", required = false) Integer threadsPerShard) {

        if (!IndexInfo.contains(index)) {
            throw new IllegalArgumentException("Index " + index + " not found!");
        }

        DistributedSearch searchService = getOrCreateDistributedSearch(index);
        
        if (threadsPerShard != null) {
            searchService.setThreadsPerShard(threadsPerShard);
        }

        List<Map<String, Object>> candidates = searchService.search(query, hits, efSearch, encoder, queryGenerator);

        Map<String, Object> queryMap = new LinkedHashMap<>();
        queryMap.put("query", new LinkedHashMap<>(Map.of("qid", qid, "text", query)));
        queryMap.put("candidates", candidates);

        return queryMap;
    }

    @RequestMapping(method = RequestMethod.GET, path = "/distributed-indexes/{index}/settings")
    public Map<String, Object> getDistributedSettings(@PathVariable("index") String index) {
        if (!IndexInfo.contains(index)) {
            throw new IllegalArgumentException("Index " + index + " not found!");
        }

        DistributedSearch service = getOrCreateDistributedSearch(index);
        return Map.of("threadsPerShard", service.getThreadsPerShard());
    }

    @RequestMapping(method = RequestMethod.POST, path = "/distributed-indexes/{index}/settings")
    public Map<String, Object> updateDistributedSettings(
            @PathVariable("index") String index,
            @RequestParam(value = "threadsPerShard", required = false) Integer threadsPerShard) {

        if (!IndexInfo.contains(index)) {
            throw new IllegalArgumentException("Index " + index + " not found!");
        }

        DistributedSearch service = getOrCreateDistributedSearch(index);
        if (threadsPerShard != null) {
            service.setThreadsPerShard(threadsPerShard);
        }

        return Map.of("status", "success");
    }

    private DistributedSearch getOrCreateDistributedSearch(String index) {
        return services.computeIfAbsent(index, k -> new DistributedSearch(k));
    }
}