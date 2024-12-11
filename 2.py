import matplotlib.pyplot as plt
import matplotlib.animation as animation
import logging
import pdb

# Configure logging
logging.basicConfig(level=logging.DEBUG, filename='animation_debug.log', 
                    filemode='w', format='%(asctime)s - %(levelname)s - %(message)s')

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
annotation = ax.text(0.02, 0.95, '', transform=ax.transAxes)

data_states = []

def update(frame):
    # Uncomment the next line to use the debugger
    # pdb.set_trace()
    
    current_data = data[frame]
    data_states.append(current_data.copy())
    
    # Log the current state
    logging.debug(f"Frame {frame}: {current_data}")
    
    # Update plot
    line.set_data(current_data['x'], current_data['y'])
    annotation.set_text(f"Frame: {frame}, Value: {current_data['value']}")
    
    return line, annotation

ani = animation.FuncAnimation(fig, update, frames=len(data), 
                              interval=500, blit=False)  # Set blit to False for easier debugging

plt.show()

# After the animation, you can further analyze `data_states` or check 'animation_debug.log'
