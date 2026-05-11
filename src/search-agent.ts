import { StateDefinition, START, END, MemorySaver, StateGraph } from "@langchain/langgraph"
import z from "zod";
import { ChatOllama } from "@langchain/ollama"
import { tavily } from "@tavily/core"
import dotenv from "dotenv"
dotenv.config();
console.log("Tavily API Key loaded:", !!process.env.TAVILY_API_KEY)  
const tavilyClient = tavily({
    apiKey: process.env.TAVILY_API_KEY || ""
})

const ollama = new ChatOllama({
    model: "mistral"
})

const SearchAgentState = z.object({
    query: z.string(),
    url: z.string().optional(),
    results: z.array(z.string()).optional(),
    thought: z.string().optional(),
    action: z.enum(["searchWeb", "ScrapeWeb", "Finish", "HumanFeedback"]).optional(),
    answer: z.string().optional(),
    humanDecision: z.enum(["approve", "retry"]).optional(),
    iterations: z.number().default(0),
})

export type SearchAgentStateType = z.infer<typeof SearchAgentState>;

const llm = ollama

async function searchWeb(state: SearchAgentStateType): Promise<Partial<SearchAgentStateType>> {
    try {
        const searchQuery = state.query;
        const response = await tavilyClient.search(searchQuery, {
            searchDepth: "basic", // ✅ "basic" | "advanced" — not "fast"
            maxResults: 5,
        });

        const results = response.results.map(r => r.content).filter(Boolean);
     console.log("Search results:", results)  
        return { results, iterations: (state.iterations ?? 0) + 1 }
    } catch (err) {
        return {
            thought: `Search error: ${err}`,
            action: "Finish",
            answer: `Sorry, I encountered an error during search. Please try again.`
        }
    }
}

async function ScrapeWeb(state: SearchAgentStateType): Promise<Partial<SearchAgentStateType>> {
    if (!state.url) {
        // No URL to scrape — fall back to search
        return {
            thought: "No URL available to scrape, falling back to search results.",
            action: "Finish"
        }
    }

    try {
        const scrapedData = await tavilyClient.crawl(state.url, {
            maxDepth: 1,
            maxBreadth: 5,
        });
  //@ts-ignore
        const results = scrapedData.results.map(r => r.content).filter(Boolean);
    console.log("Scraped results:", results)
        return { results, iterations: (state.iterations ?? 0) + 1 }
    } catch (err) {
        return {
            thought: `Scrape error: ${err}`,
            action: "Finish",
            answer: `Sorry, I encountered an error while scraping. Please try again.`
        }
    }
}


async function Reason_Node(state: SearchAgentStateType): Promise<Partial<SearchAgentStateType>> {
    const systemPrompt = `You are a search agent that answers queries using web tools.
Available actions:
- searchWeb: Search the web for the query. Use this when you need fresh information.
- ScrapeWeb: Deeply scrape a specific URL. Only use if you have a relevant URL in state.
- Finish: You have enough information to answer. Use this when results are sufficient.

Be decisive. If you have good results already, use Finish. Avoid searching more than 3 times.

IMPORTANT: Always respond with valid JSON in this exact format (and nothing else):
{
  "thought": "your reasoning here",
  "action": "searchWeb" | "ScrapeWeb" | "Finish",
  "answer": "optional answer if action is Finish"
}`

    try {
        const response = await llm.invoke([
            { role: "system", content: systemPrompt },
            {
                role: "user",
                content: `Query: ${state.query}
URL in state: ${state.url ?? "none"}
Results so far: ${state.results?.join("\n\n") ?? "none"}
Previous thought: ${state.thought ?? "none"}
Iteration count: ${state.iterations ?? 0}

What is your next action? Reply with JSON only.`
            }
        ])

        // Extract JSON from response
        const responseText = response.content
        //@ts-ignore
        const jsonMatch = responseText.match(/\{[\s\S]*\}/)
        if (!jsonMatch) {
            console.warn("No JSON found in response, raw response:", responseText)
            return {
                thought: "Could not parse LLM response",
                action: "Finish",
                answer: "I need to finish now due to parsing issues.",
            }
        }

        const parsed = JSON.parse(jsonMatch[0])
      console.log("Parsed LLM response:", parsed)
        return {
            thought: parsed.thought || "No thought",
            action: parsed.action || "Finish",
            answer: parsed.answer ?? state.answer,
        }
    } catch (error) {
        console.error(" LLM Error in Reason_Node:", error)
        return {
            thought: `LLM error: ${error}`,
            action: "Finish",
            answer: `Sorry, I encountered an error while reasoning. Please try again.`
        }
    }
}

function Observe_Node(state: SearchAgentStateType): Partial<SearchAgentStateType> {
    const results = state.results ?? [];

    if (results.length === 0) {
        return { thought: "No results found yet." }
    }

    const observation = `Observed ${results.length} result(s):\n\n${results.slice(0, 3).join("\n\n---\n\n")}`
    console.log("Observation for Reason_Node:", observation)
    return { thought: observation }
}

function Synthesize_Node(state: SearchAgentStateType): Partial<SearchAgentStateType> {
    const answer = state.answer
        ?? state.results?.[0]
        ?? "Sorry, I couldn't find any information on that topic."
    console.log("Synthesized answer for Human_Node:", answer)
    return { answer }
}

function Human_Node(state: SearchAgentStateType): Partial<SearchAgentStateType> {
    // When LangGraph resumes after interrupt, humanDecision will be
    // in the state (injected by your app via graph.invoke() with updated state).
    // We just pass it through — the conditional edge below handles routing.

    //This can be done with interruptBefore + graph.invoke()  
    return {
        humanDecision: state.humanDecision ?? "approve" 
    }
}



function reasonRouter(state: SearchAgentStateType): string {
    //I don't have infinite tokens bro just stop 
    if ((state.iterations ?? 0) >= 4) {
        console.warn("Max iterations reached, forcing finish")
        return "Synthesize_Node"
    }

    if (state.action === "searchWeb") return "searchWeb"
    if (state.action === "ScrapeWeb") return "ScrapeWeb"
    if (state.action === "Finish") return "Synthesize_Node"  


    console.warn("Unrecognized action from Reason_Node:", state.action, "Defaulting to reasoning again.")
    return "Reason_Node"
}
function humanRouter(state: SearchAgentStateType): string {
    if (state.humanDecision === "retry") {
        return "Reason_Node"
    }
    return END
}

const memory = new MemorySaver();

export const SearchAgentGraph = new StateGraph(SearchAgentState)
    .addNode("Reason_Node", Reason_Node)
    .addNode("Observe_Node", Observe_Node)
    .addNode("searchWeb", searchWeb)
    .addNode("ScrapeWeb", ScrapeWeb)
    .addNode("Synthesize_Node", Synthesize_Node)
    .addNode("Human_Node", Human_Node)

    .addEdge(START, "Reason_Node")
    .addConditionalEdges("Reason_Node", reasonRouter, {
        searchWeb: "searchWeb",
        ScrapeWeb: "ScrapeWeb",
        Synthesize_Node: "Synthesize_Node",
        Reason_Node: "Reason_Node",
    })
    .addEdge("searchWeb", "Observe_Node")
    .addEdge("ScrapeWeb", "Observe_Node")

    .addEdge("Observe_Node", "Reason_Node")

    .addEdge("Synthesize_Node", "Human_Node")
    .addConditionalEdges("Human_Node", humanRouter, {
        Reason_Node: "Reason_Node",
        [END]: END,
    })

    .compile({
        checkpointer: memory,
        interruptBefore: ["Human_Node"],
    })



    export async function runSearchAgent(query: string, threadId: string) {
    const config = { configurable: { thread_id: threadId } }

    const result = await SearchAgentGraph.invoke(
        { query, iterations: 0 },
        config
    )

    console.log("Agent answer:", result.answer)
    console.log("Waiting for human approval...")


    const humanDecision: "approve" | "retry" = "approve" // or "retry"

    // Resume by invoking again with updated state
    const finalResult = await SearchAgentGraph.invoke(
        { ...result, humanDecision },
        config
    )

    return finalResult.answer
}

import * as readline from "readline";
export async function UserInput(){
    const threadId = crypto.randomUUID() ;
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    rl.question("Enter your query: ", async (query) => {
       rl.close();
       const answer = await runSearchAgent(query, threadId);
       console.log("\nFinal Answer:", answer);
    })
}

UserInput();
