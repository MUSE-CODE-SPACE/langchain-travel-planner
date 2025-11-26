/**
 * AI Travel Planner - Frontend Application
 */

// Configuration
const API_BASE = '';
const SESSION_ID = 'session_' + Math.random().toString(36).substr(2, 9);

// State
let isLoading = false;

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const loadingOverlay = document.getElementById('loading-overlay');
const destinationsList = document.getElementById('destinations-list');
const tripContext = document.getElementById('trip-context');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadDestinations();
    chatInput.focus();
});

// Load destinations
async function loadDestinations() {
    try {
        const response = await fetch(`${API_BASE}/api/destinations`);
        const data = await response.json();

        destinationsList.innerHTML = data.destinations.map(dest => `
            <div class="destination-item" onclick="selectDestination('${dest.name}')">
                <div class="name">${dest.name}</div>
                <div class="country">${dest.country}</div>
                <div class="cost">~$${dest.avg_daily_cost}/day</div>
            </div>
        `).join('');
    } catch (error) {
        console.error('Failed to load destinations:', error);
        destinationsList.innerHTML = `
            <div class="destination-item" onclick="selectDestination('Tokyo')">
                <div class="name">Tokyo</div>
                <div class="country">Japan</div>
                <div class="cost">~$150/day</div>
            </div>
            <div class="destination-item" onclick="selectDestination('Paris')">
                <div class="name">Paris</div>
                <div class="country">France</div>
                <div class="cost">~$180/day</div>
            </div>
            <div class="destination-item" onclick="selectDestination('Seoul')">
                <div class="name">Seoul</div>
                <div class="country">South Korea</div>
                <div class="cost">~$100/day</div>
            </div>
            <div class="destination-item" onclick="selectDestination('Barcelona')">
                <div class="name">Barcelona</div>
                <div class="country">Spain</div>
                <div class="cost">~$130/day</div>
            </div>
            <div class="destination-item" onclick="selectDestination('Bangkok')">
                <div class="name">Bangkok</div>
                <div class="country">Thailand</div>
                <div class="cost">~$60/day</div>
            </div>
        `;
    }
}

// Select destination
function selectDestination(destination) {
    sendQuickMessage(`I want to visit ${destination}`);
}

// Send quick message
function sendQuickMessage(message) {
    chatInput.value = message;
    sendMessage();
}

// Handle key press
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

// Send message
async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || isLoading) return;

    // Add user message
    addMessage(message, 'user');
    chatInput.value = '';

    // Show loading
    setLoading(true);

    try {
        const response = await fetch(`${API_BASE}/api/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                session_id: SESSION_ID
            })
        });

        const data = await response.json();

        if (data.error) {
            addMessage('Sorry, something went wrong. Please try again.', 'assistant');
        } else {
            addMessage(data.response, 'assistant');
            updateContext(data.context);
        }
    } catch (error) {
        console.error('Chat error:', error);
        // Simulate response for demo
        simulateResponse(message);
    }

    setLoading(false);
}

// Simulate response (for demo without backend)
function simulateResponse(message) {
    const lowerMessage = message.toLowerCase();

    let response = '';

    if (lowerMessage.includes('tokyo') || lowerMessage.includes('japan')) {
        response = `**Tokyo** is an excellent choice! Here's what makes it special:

- **Best Time to Visit**: Spring (March-May) for cherry blossoms or Autumn (Sept-Nov) for pleasant weather
- **Average Daily Cost**: ~$150 USD
- **Top Attractions**: Senso-ji Temple, Shibuya Crossing, Meiji Shrine, Tokyo Skytree

**Sample Day in Tokyo:**
- 09:00 - Visit Senso-ji Temple in Asakusa
- 12:00 - Lunch at Tsukiji Outer Market
- 14:00 - Explore teamLab Borderless
- 18:00 - Experience Shibuya Crossing
- 19:30 - Dinner at a local izakaya

Would you like me to:
1. Find hotels in Tokyo?
2. Suggest more activities?
3. Create a full itinerary?`;
        updateDemoContext({ destination: 'Tokyo', interests: ['cultural', 'food'] });
    }
    else if (lowerMessage.includes('paris') || lowerMessage.includes('france')) {
        response = `**Paris** - the City of Light awaits! Here's your guide:

- **Best Time to Visit**: Spring (April-June) or Autumn (Sept-Oct)
- **Average Daily Cost**: ~$180 USD
- **Top Attractions**: Eiffel Tower, Louvre Museum, Notre-Dame, Montmartre

**Must-Do Experiences:**
- Sunrise at the Eiffel Tower
- Croissant breakfast at a local boulangerie
- Seine River sunset cruise
- Wine tasting in Le Marais

What would you like to explore next?`;
        updateDemoContext({ destination: 'Paris', interests: ['cultural', 'food'] });
    }
    else if (lowerMessage.includes('seoul') || lowerMessage.includes('korea')) {
        response = `**Seoul** is perfect for K-culture lovers! Here's what awaits:

- **Best Time to Visit**: Spring or Autumn
- **Average Daily Cost**: ~$100 USD
- **Top Attractions**: Gyeongbokgung Palace, Bukchon Hanok Village, Myeongdong

**Korean Experiences:**
- Traditional hanbok wearing at palaces
- Korean BBQ dinner
- K-pop dance class in Hongdae
- Jjimjilbang (Korean spa) experience

Ready to plan your Seoul adventure?`;
        updateDemoContext({ destination: 'Seoul', interests: ['cultural', 'food'] });
    }
    else if (lowerMessage.includes('hotel') || lowerMessage.includes('stay') || lowerMessage.includes('accommodation')) {
        response = `Here are accommodation options I found:

**Luxury ($$$$)**
- Grand Hotel Downtown - $250/night, Rating: 4.8
- Features: Spa, pool, rooftop restaurant

**Mid-Range ($$$)**
- City Center Hotel - $120/night, Rating: 4.5
- Features: Gym, breakfast included, central location

**Budget ($$)**
- Cozy Guesthouse - $45/night, Rating: 4.3
- Features: WiFi, shared kitchen, social atmosphere

Which style suits your trip best?`;
    }
    else if (lowerMessage.includes('budget') || lowerMessage.includes('cost')) {
        response = `**Budget Estimate for Your Trip**

Based on a moderate budget for 5 days:

| Category | Cost |
|----------|------|
| Accommodation | $600 |
| Food & Dining | $250 |
| Activities | $150 |
| Transportation | $100 |
| Miscellaneous | $100 |

**Total: ~$1,200** (excluding flights)

**Money-Saving Tips:**
- Book attractions online in advance
- Eat at local restaurants vs tourist spots
- Use public transportation
- Visit free attractions and parks

Would you like a more detailed breakdown?`;
    }
    else if (lowerMessage.includes('itinerary') || lowerMessage.includes('plan')) {
        response = `# Your 5-Day Adventure Itinerary

## Day 1: Arrival & Exploration
**09:00** - Check-in and freshen up
**12:00** - Local lunch in the neighborhood
**14:00** - Walking tour of the city center
**19:00** - Welcome dinner at recommended restaurant

## Day 2: Cultural Immersion
**09:00** - Visit iconic temple/landmark
**12:30** - Traditional cuisine lunch
**14:30** - Museum exploration
**18:00** - Sunset viewpoint
**20:00** - Local dining experience

## Day 3: Local Experiences
**10:00** - Cooking class
**14:00** - Market exploration
**17:00** - Neighborhood walk
**19:30** - Street food dinner

## Day 4: Adventure Day
**08:00** - Day trip excursion
**18:00** - Return and rest
**20:00** - Farewell dinner

## Day 5: Departure
**10:00** - Last-minute shopping
**12:00** - Checkout and departure

---

**Estimated Budget: $1,200**
**Packing Essentials:** Comfortable shoes, adapter, light layers

Want me to customize any of these days?`;
    }
    else if (lowerMessage.includes('activity') || lowerMessage.includes('things to do') || lowerMessage.includes('attraction')) {
        response = `Here are top activities I recommend:

**Cultural & Historical**
- Ancient Temple Visit (2 hrs, Free)
- Historical Walking Tour (3 hrs, $25)
- Traditional Craft Workshop (2 hrs, $45)

**Food & Culinary**
- Street Food Tour (3 hrs, $60)
- Cooking Class (4 hrs, $80)
- Market Experience (2 hrs, Free)

**Entertainment**
- Night Market Visit (3 hrs, Free)
- Traditional Performance (2 hrs, $40)

**Nature & Outdoors**
- Park & Gardens Tour (2 hrs, Free)
- Scenic Viewpoint Visit (1 hr, $15)

Which type of activities interest you most?`;
    }
    else if (lowerMessage.includes('weather')) {
        response = `**Weather Forecast**

📍 Current conditions for your destination:

- **Temperature**: 18°C - 24°C
- **Condition**: Partly Cloudy
- **Humidity**: 55%
- **Rain Chance**: 20%

**Packing Recommendation:**
Bring layers as temperatures vary throughout the day. A light jacket and umbrella are essential. Comfortable walking shoes are a must!

Best time to visit outdoor attractions: Morning or late afternoon.`;
    }
    else if (lowerMessage.includes('recommend') || lowerMessage.includes('suggest') || lowerMessage.includes('where')) {
        response = `Based on popular choices, here are my top destination recommendations:

**1. Tokyo, Japan**
Perfect for: Culture lovers, foodies, tech enthusiasts
Best time: Spring (cherry blossoms) or Autumn
Budget: ~$150/day

**2. Barcelona, Spain**
Perfect for: Architecture fans, beach lovers, nightlife
Best time: May-June or September
Budget: ~$130/day

**3. Seoul, South Korea**
Perfect for: K-culture fans, foodies, shoppers
Best time: Spring or Autumn
Budget: ~$100/day

**4. Bali, Indonesia**
Perfect for: Relaxation, spiritual retreats, beaches
Best time: April-October
Budget: ~$70/day

What type of experience are you looking for? I can narrow down the options!`;
    }
    else {
        response = `I'd be happy to help with your travel planning! Here's what I can do:

**Destination Discovery**
- Recommend places based on your interests
- Compare destinations by budget and season

**Trip Planning**
- Find hotels and accommodations
- Suggest activities and attractions
- Create detailed itineraries

**Practical Help**
- Calculate trip budgets
- Check weather forecasts
- Provide packing lists and tips

Tell me more about your dream trip! For example:
- "I want to visit Tokyo for 5 days"
- "Recommend a beach destination"
- "What's the budget for Paris?"`;
    }

    addMessage(response, 'assistant');
}

// Update demo context
function updateDemoContext(context) {
    let html = '';

    if (context.destination) {
        html += `<div class="context-item"><span class="label">Destination</span><span class="value">${context.destination}</span></div>`;
    }
    if (context.duration_days) {
        html += `<div class="context-item"><span class="label">Duration</span><span class="value">${context.duration_days} days</span></div>`;
    }
    if (context.budget_level) {
        html += `<div class="context-item"><span class="label">Budget</span><span class="value">${context.budget_level}</span></div>`;
    }
    if (context.travelers) {
        html += `<div class="context-item"><span class="label">Travelers</span><span class="value">${context.travelers}</span></div>`;
    }
    if (context.interests && context.interests.length > 0) {
        html += `<div class="context-item"><span class="label">Interests</span><span class="value">${context.interests.join(', ')}</span></div>`;
    }

    if (html) {
        tripContext.innerHTML = html;
    }
}

// Update context from API
function updateContext(context) {
    if (!context || Object.keys(context).length === 0) {
        tripContext.innerHTML = '<p class="empty-state">Start chatting to plan your trip!</p>';
        return;
    }

    updateDemoContext(context);
}

// Add message to chat
function addMessage(content, role) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    // Parse markdown-like formatting
    let formattedContent = content
        // Headers
        .replace(/^### (.+)$/gm, '<h3>$1</h3>')
        .replace(/^## (.+)$/gm, '<h2>$1</h2>')
        .replace(/^# (.+)$/gm, '<h1>$1</h1>')
        // Bold
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        // Italic
        .replace(/\*(.+?)\*/g, '<em>$1</em>')
        // Lists
        .replace(/^- (.+)$/gm, '<li>$1</li>')
        // Tables
        .replace(/\|(.+)\|/g, (match, content) => {
            const cells = content.split('|').map(c => c.trim());
            return '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
        })
        // Line breaks
        .replace(/\n\n/g, '</p><p>')
        .replace(/\n/g, '<br>');

    // Wrap lists
    formattedContent = formattedContent.replace(/(<li>.*<\/li>)+/g, '<ul>$&</ul>');

    // Wrap tables
    formattedContent = formattedContent.replace(/(<tr>.*<\/tr>)+/g, '<table>$&</table>');

    contentDiv.innerHTML = `<p>${formattedContent}</p>`;

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Set loading state
function setLoading(loading) {
    isLoading = loading;
    loadingOverlay.classList.toggle('hidden', !loading);
    sendBtn.disabled = loading;
    chatInput.disabled = loading;
}
