import boto3
import json

class BedrockSearchEngine:
    def __init__(self, model_id="us.anthropic.claude-3-sonnet-20240229-v1:0"):
        """
        Initialize Bedrock search engine with specified model
        """
        self.bedrock = boto3.client(service_name="bedrock-runtime")
        self.model_id = model_id

    def search(self, query):
        """
        Perform search by sending query to Bedrock
        """
        try:
            # Prepare payload for model invocation
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"""
                                You are a comprehensive search engine. 
                                Provide a detailed, informative answer to the following query:
                                {query}
                                """
                            }
                        ]
                    }
                ]
            }

            # Convert payload to JSON
            body = json.dumps(payload)

            # Invoke the model
            response = self.bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json"
            )

            # Parse the response
            response_body = json.loads(response.get("body").read())
            
            # Extract response text
            content = response_body.get("content", [])
            if content and len(content) > 0:
                return content[0].get("text", "No response generated.")
            else:
                return "No content found or response structure is unexpected."

        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Create search engine instance
    search_engine = BedrockSearchEngine()

    # Interactive search loop
    while True:
        # Get user query
        query = input("\nWhat would you like to search? (or 'exit' to quit): ")
        
        # Check for exit
        if query.lower() == 'exit':
            break
        
        # Perform search and print results
        print("\n--- Search Result ---")
        result = search_engine.search(query)
        print(result)

if __name__ == "__main__":
    main()