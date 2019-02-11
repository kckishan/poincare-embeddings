import torch
import plotly.offline as plt
import plotly.graph_objs as go

def generate_report(model,data):
    print('Generating report...')
  
    x,y = zip(*enumerate(model.log))
  
    trace = go.Scatter(x = x,
                       y = y
                      )
    layout = go.Layout(
                 yaxis=dict(
                      title= 'Loss',
                      ticklen= 5,
                      gridwidth= 2,
                      ),
                 xaxis=dict(
                      title= 'Epoch',
                      ticklen= 5,
                      gridwidth= 2,
                     ))
  
    fig = go.Figure([trace],layout=layout)
    plt.plot(fig,filename='demo/log_loss.html')
  
  
  
    with open('data/wordnet/mammal_hierarchy.tsv','r') as f:
        edgelist = [line.strip().split('\t') for line in f.readlines()]
        
    vis = model.embedding.weight.data.numpy()
  
  
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=0.5,color='#888'),
        hoverinfo='none',
        mode='lines')
  
    xs = []
    ys = []
    for s0,s1 in edgelist:
        x0, y0 = vis[data.item2id[s0]]
        x1, y1 = vis[data.item2id[s1]]
  
        xs.extend(tuple([x0, x1, None]))
        ys.extend(tuple([y0, y1, None]))
  
    edge_trace['x'] = xs
    edge_trace['y'] = ys
  
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=False,
            reversescale=True,
            color='#8b9dc3',
            size=2)
        )
  
    xs = []
    ys = []
    names = []
    for name in data.items:
        x, y = vis[data.item2id[name]]
        xs.extend(tuple([x]))
        ys.extend(tuple([y]))
        names.extend(tuple([name.split('.')[0]]))
  
    node_trace['x'] = xs 
    node_trace['y'] = ys
        
    node_trace['text'] = names 
  
    display_list = ['placental.n.01',
     'primate.n.02',
     'mammal.n.01',
     'carnivore.n.01',
     'canine.n.02',
     'dog.n.01',
     'pug.n.01',
     'homo_erectus.n.01',
     'homo_sapiens.n.01',
     'terrier.n.01',
     'rodent.n.01',
     'ungulate.n.01',
     'odd-toed_ungulate.n.01',
     'even-toed_ungulate.n.01',
     'monkey.n.01',
     'cow.n.01',
     'welsh_pony.n.01',
     'feline.n.01',
     'cheetah.n.01',
     'mouse.n.01']
  
    label_trace = go.Scatter(
        x=[],
        y=[],
        mode='text',
        text=[],
        textposition='top center',
        textfont=dict(
            family='sans serif',
            size=13,
            color = "#000000"
        )
    )
  
    for name in display_list:
        x,y = vis[data.item2id[name]]
        label_trace['x'] += tuple([x])
        label_trace['y'] += tuple([y])
        label_trace['text'] += tuple([name.split('.')[0]])
  
  
  
    fig = go.Figure(data=[edge_trace, node_trace,label_trace],
                 layout=go.Layout(
                    title='Poincare Embedding of mammals subset of WordNet',
                    width=700,
                    height=700,
                    titlefont=dict(size=16),
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
  
    plt.plot(fig, filename='demo/poincare embedding.html')
  
    print('report is saves as .html files in demo folder.')

if __name__ == '__main__':
    import torch

    model = torch.load('demo/model.pt')
    data = torch.load('demo/data.pt')

    generate_report(model,data)