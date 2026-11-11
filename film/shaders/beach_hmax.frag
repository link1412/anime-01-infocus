#version 410
// For the ray march: the highest thing in each block of rows along an azimuth.
in vec2 uv; out vec4 o;
uniform sampler2D uR0;
uniform int uBlock;
uniform ivec2 uNS;
void main(){
  ivec2 p=ivec2(gl_FragCoord.xy);
  float m=-1e3;
  for(int j=-1;j<=uBlock;j++){
    int y=p.y*uBlock+j;
    if(y<0||y>=uNS.y) continue;
    for(int i=-1;i<=1;i+=1){
      int x=clamp(p.x+i,0,uNS.x-1);
      m=max(m,texelFetch(uR0,ivec2(x,y),0).r);
    }
  }
  o=vec4(m,0.,0.,1.);
}
