async def some_method(self, event):
    await self.send({
        "type": "websocket.send",
        "text": event["message"]
    })