import { tool } from "@langchain/core/tools";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import { MemorySaver } from "@langchain/langgraph";
import { ChatAnthropic } from "@langchain/anthropic";
import { z } from "zod";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";

export default async function WeatherAgent() {
    // Define system prompt
    const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

    // Define tools
    const getWeather = tool(
        async ({ city }: { city: string }) => `It's always sunny in ${city}!`,
        {
            name: "get_weather_for_location",
            description: "Get the weather for a given city",
            schema: z.object({
                city: z.string(),
            }),
        }
    );

    const getUserLocation = tool(
        async () => {
            // Simulating user location lookup
            return "Florida";
        },
        {
            name: "get_user_location",
            description: "Retrieve user location",
            schema: z.object({}),
        }
    );

    // Configure model
    const model = new ChatAnthropic({
        model: "claude-sonnet-4-5-20250929",
        temperature: 0,
    });

    // Set up memory
    const checkpointer = new MemorySaver();

    // Create agent
    const agent = createReactAgent({
        llm: model,
        tools: [getUserLocation, getWeather],
        checkpointSaver: checkpointer,
    });

    // Run agent
    // `thread_id` is a unique identifier for a given conversation.
    const config = {
        configurable: { thread_id: "1" },
    };

    const response = await agent.invoke(
        {
            messages: [
                new SystemMessage(systemPrompt),
                new HumanMessage("what is the weather outside?"),
            ],
        },
        config
    );
    console.log("First response:", response.messages[response.messages.length - 1].content);

    // Note that we can continue the conversation using the same `thread_id`.
    const thankYouResponse = await agent.invoke(
        { messages: [new HumanMessage("thank you!")] },
        config
    );
    console.log("Second response:", thankYouResponse.messages[thankYouResponse.messages.length - 1].content);

    return <div>Weather Agent executed successfully. Check console for output.</div>;
}