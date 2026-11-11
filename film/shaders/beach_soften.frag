#version 410
// Where one section of a bore ends and the next begins the solver has a wall one cell wide, running out to
// sea along the column. No water stands like that. Only for the renderer, the surface across such a wall is
// eased sideways over about twice its height; fronts that face the camera are left as they are.
in vec2 uv; out vec4 o;
uniform sampler2D uR0;
uniform vec4 uGrid;
const int TAPS=8;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy),top=textureSize(uR0,0)-1;
  vec4 c=texelFetch(uR0,p,0);
  o=c;
  float r=exp(uGrid.z+(float(p.y)+.5)*uGrid.w);
  float cl=r*uGrid.y,cr=r*uGrid.w;
  if(r<14.) return;
  // over about a metre, so that a slanting front is not mistaken for the end of one
  int kx=int(clamp(.8/cl,1.,10.)),ky=int(clamp(.8/cr,1.,10.));
  float eL=texelFetch(uR0,ivec2(max(p.x-kx,0),p.y),0).r,eR=texelFetch(uR0,ivec2(min(p.x+kx,top.x),p.y),0).r;
  float eN=texelFetch(uR0,ivec2(p.x,min(p.y+ky,top.y)),0).r,eQ=texelFetch(uR0,ivec2(p.x,max(p.y-ky,0)),0).r;
  float rise=abs(eR-eL);
  float sL=rise/(2.*float(kx)*cl),sR=abs(eN-eQ)/(2.*float(ky)*cr);
  float w=smoothstep(.1,.32,sL)*smoothstep(.75,1.7,sL/(sR+.03));
  if(w<=0.) return;
  float half_=clamp(rise*2.4,.5,3.2);        // metres either side
  vec2 acc=vec2(0.); float W=0.;
  vec2 tx=1./vec2(textureSize(uR0,0));
  vec2 pc=(vec2(p)+.5)*tx;
  for(int i=-TAPS;i<=TAPS;i++){
    float f=float(i)/float(TAPS),g=(1.-f*f)*(1.-f*f);
    vec2 a=textureLod(uR0,vec2(pc.x+f*half_/cl*tx.x,pc.y),0.).rg;
    acc+=a*g; W+=g;
  }
  o.rg=mix(c.rg,acc/W,w);
}
